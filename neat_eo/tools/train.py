import os
import math
import uuid
from tqdm import tqdm

import torch
import torch.optim
import torch.backends.cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import neat_eo as neo
from neat_eo.core import load_config, load_module, check_model, check_channels, check_classes, Logs
from neat_eo.tiles import tiles_from_csv
from neat_eo.metrics.core import Metrics
from neat_eo.tools.dataset import compute_classes_weights


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser("train", help="Trains a model on a dataset", formatter_class=formatter_class)
    parser.add_argument("--config", type=str, help="path to config file [required, if no global config setting]")

    data = parser.add_argument_group("Dataset")
    data.add_argument("--train_dataset", type=str, help="train dataset path [needed for train]")
    data.add_argument("--eval_dataset", type=str, help="eval dataset path [needed for eval]")
    data.add_argument("--cover", type=str, help="path to csv tiles cover file, to filter tiles dataset on [optional]")
    data.add_argument("--classes_weights", type=str, help="classes weights separated with comma or 'auto' [optional]")
    data.add_argument("--tiles_weights", type=str, help="path to csv tiles cover file, with specials weights on [optional]")
    data.add_argument("--loader", type=str, help="dataset loader name [if set override config file value]")

    hp = parser.add_argument_group("Hyper Parameters [if set override config file value]")
    hp.add_argument("--bs", type=int, help="batch size")
    hp.add_argument("--lr", type=float, help="learning rate")
    hp.add_argument("--ts", type=str, help="tile size")
    hp.add_argument("--nn", type=str, help="neurals network name")
    hp.add_argument("--encoder", type=str, help="encoder name")
    hp.add_argument("--optimizer", type=str, help="optimizer name")
    hp.add_argument("--loss", type=str, help="model loss")

    mt = parser.add_argument_group("Training")
    mt.add_argument("--epochs", type=int, help="number of epochs to train")
    mt.add_argument("--resume", action="store_true", help="resume model training, if set imply to provide a checkpoint")
    mt.add_argument("--checkpoint", type=str, help="path to a model checkpoint. To fine tune or resume a training")
    mt.add_argument("--workers", type=int, help="number of pre-processing images workers, per GPU [default: batch size]")

    out = parser.add_argument_group("Output")
    out.add_argument("--saving", type=int, default=1, help="number of epochs beetwen checkpoint saving [default: 1]")
    out.add_argument("--out", type=str, required=True, help="output directory path to save checkpoint and logs [required]")

    parser.set_defaults(func=main)


def main(args):
    config = load_config(args.config)
    args.out = os.path.expanduser(args.out)
    args.cover = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))] if args.cover else None
    if args.classes_weights:
        try:
            args.classes_weights = list(map(float, args.classes_weights.split(",")))
        except:
            assert args.classes_weights == "auto", "invalid --classes_weights value"
    else:
        args.classes_weights = [classe["weight"] for classe in config["classes"]]

    args.tiles_weights = (
        [(tile, weight) for tile, weight in tiles_from_csv(os.path.expanduser(args.tiles_weights), extra_columns=True)]
        if args.tiles_weights
        else None
    )

    config["model"]["loader"] = args.loader if args.loader else config["model"]["loader"]
    config["model"]["ts"] = tuple(map(int, args.ts.split(","))) if args.ts else config["model"]["ts"]
    config["model"]["nn"] = args.nn if args.nn else config["model"]["nn"]
    config["model"]["encoder"] = args.encoder if args.encoder else config["model"]["encoder"]
    config["train"]["bs"] = args.bs if args.bs else config["train"]["bs"]
    config["train"]["loss"] = args.loss if args.loss else config["train"]["loss"]
    config["train"]["optimizer"]["name"] = args.optimizer if args.optimizer else config["train"]["optimizer"]["name"]
    config["train"]["optimizer"]["lr"] = args.lr if args.lr else config["train"]["optimizer"]["lr"]
    check_classes(config)
    check_channels(config)
    check_model(config)

    log = Logs(os.path.join(args.out, "log"))

    assert torch.cuda.is_available(), "No GPU support found. Check CUDA and NVidia Driver install."
    assert torch.distributed.is_nccl_available(), "No NCCL support found. Check your PyTorch install."
    world_size = torch.cuda.device_count() if args.train_dataset else 1

    args.workers = min(config["train"]["bs"] if not args.workers else args.workers, math.floor(os.cpu_count() / world_size))
    assert args.eval_dataset or args.train_dataset, "Provide at least one dataset"

    if args.eval_dataset and not args.train_dataset and not args.checkpoint:
        log.log("\n\nNOTICE: No Checkpoint provided for eval only. Seems peculiar.\n\n")

    log.log("neo train/eval on {} GPUs, with {} workers/GPU".format(world_size, args.workers))
    log.log("---")

    loader = load_module("neat_eo.loaders.{}".format(config["model"]["loader"].lower()))

    train_dataset = None
    if args.train_dataset:
        assert os.path.isdir(os.path.expanduser(args.train_dataset)), "--train_dataset path is not a directory"
        train_dataset = getattr(loader, config["model"]["loader"])(
            config, config["model"]["ts"], args.train_dataset, args.cover, args.tiles_weights, "train"
        )
        assert len(train_dataset), "Empty or Invalid --train_dataset content"
        shape_in = train_dataset.shape_in
        shape_out = train_dataset.shape_out
        log.log("\nDataSet Training:        {}".format(args.train_dataset))

        if args.classes_weights == "auto":
            args.classes_weights = compute_classes_weights(args.train_dataset, config["classes"], args.cover, os.cpu_count())

    eval_dataset = None
    if args.eval_dataset:
        assert os.path.isdir(os.path.expanduser(args.eval_dataset)), "--eval_dataset path is not a directory"
        eval_dataset = getattr(loader, config["model"]["loader"])(
            config, config["model"]["ts"], args.eval_dataset, args.cover, args.tiles_weights, "eval"
        )
        assert len(eval_dataset), "Empty or Invalid --eval_dataset content"
        shape_in = eval_dataset.shape_in
        shape_out = eval_dataset.shape_out
        log.log("DataSet Eval:            {}".format(args.eval_dataset))

        if not args.train_dataset and args.classes_weights == "auto":
            args.classes_weights = compute_classes_weights(args.eval_dataset, config["classes"], args.cover, os.cpu_count())

    log.log("\n--- Input tensor")
    num_channel = 1  # 1-based numerotation
    for channel in config["channels"]:
        for band in channel["bands"]:
            log.log("Channel {}:\t\t {} - (band:{})".format(num_channel, channel["name"], band))
            num_channel += 1

    log.log("\n--- Output Classes ---")
    for c, classe in enumerate(config["classes"]):
        log.log("Class {}:\t\t {} ({:.2f})".format(c, classe["title"], args.classes_weights[c]))

    log.log("\n--- Model ---")
    for hp in config["model"]:
        log.log("{}{}".format(hp.ljust(25, " "), config["model"][hp]))

    lock_file = os.path.abspath(os.path.join(args.out, str(uuid.uuid1())))
    mp.spawn(
        gpu_worker,
        nprocs=world_size,
        args=(world_size, lock_file, train_dataset, eval_dataset, shape_in, shape_out, args, config),
    )
    if os.path.exists(lock_file):
        os.remove(lock_file)


def gpu_worker(rank, world_size, lock_file, train_dataset, eval_dataset, shape_in, shape_out, args, config):

    log = Logs(os.path.join(args.out, "log")) if rank == 0 else None
    csv_train = open(os.path.join(args.out, "train.csv"), mode="a") if train_dataset and rank == 0 else None
    csv_eval = open(os.path.join(args.out, "eval.csv"), mode="a") if eval_dataset and rank == 0 else None

    dist.init_process_group(backend="nccl", init_method="file://" + lock_file, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    bs = config["train"]["bs"]

    if train_dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=False, drop_last=True, num_workers=args.workers, sampler=sampler
        )
    else:
        train_loader = None

    if eval_dataset:
        eval_loader = DataLoader(eval_dataset, batch_size=bs, shuffle=False, drop_last=True, num_workers=args.workers)
    else:
        eval_loader = None

    nn_module = load_module("neat_eo.nn.{}".format(config["model"]["nn"].lower()))
    nn = getattr(nn_module, config["model"]["nn"])(
        shape_in, shape_out, config["model"]["encoder"].lower(), config["train"]
    ).cuda(rank)
    nn = DistributedDataParallel(nn, device_ids=[rank], find_unused_parameters=True)

    if train_dataset:
        optimizer_params = {key: value for key, value in config["train"]["optimizer"].items() if key != "name"}
        optimizer = getattr(torch.optim, config["train"]["optimizer"]["name"])(nn.parameters(), **optimizer_params)

        if rank == 0:
            log.log("\n--- Train ---")
            for hp in config["train"]:
                if hp == "da":
                    da = config["train"]["da"]["name"]
                    dap = config["train"]["da"]["p"]
                    log.log("{}{} ({:.2f})".format("da".ljust(25, " "), da, dap))
                elif hp == "metrics":
                    log.log("{}{}".format(hp.ljust(25, " "), set(config["train"][hp])))  # aesthetic
                elif hp != "optimizer":
                    log.log("{}{}".format(hp.ljust(25, " "), config["train"][hp]))

            log.log("{}{}".format("optimizer".ljust(25, " "), config["train"]["optimizer"]["name"]))
            for k, v in optimizer.state_dict()["param_groups"][0].items():
                if k != "params":
                    log.log(" - {}{}".format(k.ljust(25 - 3, " "), v))

    resume = 0
    if args.checkpoint:
        chkpt = torch.load(os.path.expanduser(args.checkpoint), map_location="cuda:{}".format(rank))
        assert nn.module.version == chkpt["model_version"], "Model Version mismatch"
        nn.load_state_dict(chkpt["state_dict"])

        if rank == 0:
            log.log("\n--- Using Checkpoint ---")
            log.log("Path:\t\t {}".format(args.checkpoint))
            log.log("UUID:\t\t {}".format(chkpt["uuid"]))

        if args.resume:
            optimizer.load_state_dict(chkpt["optimizer"])
            resume = chkpt["epoch"]
            assert resume < args.epochs, "Epoch asked, already reached by the given checkpoint"

    loss_module = load_module("neat_eo.losses.{}".format(config["train"]["loss"].lower()))
    criterion = getattr(loss_module, config["train"]["loss"])().cuda(rank)

    if eval_dataset and not train_dataset:
        do_epoch(rank, eval_loader, config, args.classes_weights, log, csv_eval, nn, criterion, "eval", 1)
        dist.destroy_process_group()
        return

    for epoch in range(resume + 1, args.epochs + 1):  # 1-N based

        if train_dataset:
            if rank == 0:
                log.log("\n---\nEpoch: {}/{}\n".format(epoch, args.epochs))

            sampler.set_epoch(epoch)  # https://github.com/pytorch/pytorch/issues/31232
            do_epoch(
                rank, train_loader, config, args.classes_weights, log, csv_train, nn, criterion, "train", epoch, optimizer
            )

            if rank == 0:
                UUID = uuid.uuid1()
                states = {
                    "uuid": UUID,
                    "model_version": nn.module.version,
                    "producer_name": "Neat-EO.pink",
                    "producer_version": neo.__version__,
                    "model_licence": "MIT",
                    "domain": "pink.Neat-EO",  # reverse-DNS
                    "doc_string": nn.module.doc_string,
                    "shape_in": shape_in,
                    "shape_out": shape_out,
                    "state_dict": nn.state_dict(),
                    "epoch": epoch,
                    "nn": config["model"]["nn"],
                    "encoder": config["model"]["encoder"],
                    "optimizer": optimizer.state_dict(),
                    "loader": config["model"]["loader"],
                }
                checkpoint_path = os.path.join(args.out, "checkpoint-{:05d}.pth".format(epoch))
                if epoch == args.epochs or not (epoch % args.saving):
                    log.log("\n--- Saving Checkpoint ---")
                    log.log("Path:\t\t {}".format(checkpoint_path))
                    log.log("UUID:\t\t {}\n".format(UUID))
                    torch.save(states, checkpoint_path)

            dist.barrier()

        if eval_dataset:
            do_epoch(rank, eval_loader, config, args.classes_weights, log, csv_eval, nn, criterion, "eval", epoch)

    dist.destroy_process_group()


def do_epoch(rank, loader, config, classes_weights, log, csv, nn, criterion, mode, epoch, optimizer=None):
    def _do_epoch():
        num_samples = 0
        running_loss = 0.0
        metrics = Metrics(config["train"]["metrics"], config["classes"], config=config)

        unit = "Batch/GPU" if mode == "train" else "Batch"
        assert len(loader), "Empty or Inconsistent DataSet"

        dataloader = tqdm(loader, desc=mode.title(), unit=unit, ascii=True) if rank == 0 else loader

        for images, masks, tiles, tiles_weights in dataloader:
            images = images.cuda(rank, non_blocking=True)
            masks = masks.cuda(rank, non_blocking=True)

            num_samples += int(images.size(0))

            # Forward
            outputs = nn(images)
            loss = criterion(outputs, masks, classes_weights, tiles_weights, config)
            running_loss += loss.item()

            # Backward
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Metrics
            if mode == "eval" and rank == 0:
                for mask, output in zip(masks, outputs):
                    metrics.add(mask, output)

        assert num_samples > 0, "DataSet inconsistencies"

        if rank == 0:
            assert log and csv, "Unable to log"
            log.log("{}{:.3f}".format("Loss:".ljust(25, " "), running_loss / num_samples))
            csv_header = ['"Epoch"', '"Loss"'] if epoch == 1 else None
            csv_line = [str(epoch)]
            csv_line.append("{:.4f}".format(running_loss / num_samples))
            if mode == "eval":
                log.log("\n{}  μ\t   σ".format(" ".ljust(25, " ")))
                for c, classe in enumerate(config["classes"]):
                    if classe["weight"] != 0.0 and classe["color"] != "transparent":
                        for k, v in metrics.get()[c].items():
                            log.log("{}{:.3f}\t {:.3f}".format((classe["title"] + " " + k).ljust(25, " "), v["μ"], v["σ"]))
                            csv_header.append('"{} μ{}"'.format(classe["title"], k)) if epoch == 1 else None
                            csv_header.append('"{} σ{}"'.format(classe["title"], k)) if epoch == 1 else None
                            csv_line.append("{:.3f}".format(v["μ"]))
                            csv_line.append("{:.3f}".format(v["σ"]))

            if epoch == 1:
                csv.write(",".join(csv_header) + os.linesep)
            csv.write(",".join(csv_line) + os.linesep)
            csv.flush()

    if mode == "train":
        nn.train()
        _do_epoch()

    if mode == "eval":
        nn.eval()
        with torch.no_grad():
            _do_epoch()
