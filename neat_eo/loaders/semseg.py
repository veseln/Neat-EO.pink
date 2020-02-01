"""PyTorch-compatible datasets. Cf: https://pytorch.org/docs/stable/data.html """

import os
import numpy as np
import torch.utils.data

from neat_eo.tiles import tiles_from_dir, tile_image_from_file, tile_label_from_file, tile_image_buffer
from neat_eo.da.core import to_tensor


class SemSeg(torch.utils.data.Dataset):
    def __init__(self, config, ts, root, cover=None, tiles_weights=None, mode=None, metatiles=False):
        super().__init__()

        self.mode = mode
        self.cover = cover
        self.config = config
        self.tiles_weights = tiles_weights
        self.metatiles = metatiles
        self.root = root
        self.data_augmentation = (
            True if "da" in self.config["train"].keys() and self.config["train"]["da"]["p"] > 0.0 else False
        )

        assert mode in ["train", "eval", "predict"]

        num_channels = 0
        self.tiles = {}
        for channel in config["channels"]:
            path = os.path.join(self.root, channel["name"])
            self.tiles[channel["name"]] = [(tile, path) for tile, path in tiles_from_dir(path, cover=cover, xyz_path=True)]
            num_channels += len(channel["bands"])

        self.shape_in = (num_channels,) + tuple(ts)  # C,W,H
        self.shape_out = (len(config["classes"]),) + tuple(ts)  # C,W,H

        if self.mode in ["train", "eval"]:
            path = os.path.join(self.root, "labels")
            self.tiles["labels"] = [(tile, path) for tile, path in tiles_from_dir(path, cover=cover, xyz_path=True)]

            for channel in config["channels"]:  # Order images and labels accordingly
                self.tiles[channel["name"]].sort(key=lambda tile: tile[0])
            self.tiles["labels"].sort(key=lambda tile: tile[0])

        assert len(self.tiles), "Empty Dataset"

    def __len__(self):
        return len(self.tiles[self.config["channels"][0]["name"]])

    def __getitem__(self, i):

        tile = None
        mask = None
        image = None

        for channel in self.config["channels"]:

            image_channel = None
            tile, path = self.tiles[channel["name"]][i]
            bands = None if not channel["bands"] else channel["bands"]

            if self.metatiles:
                image_channel = tile_image_buffer(tile, self.tiles[self.config["channels"][0]["name"]], bands)
            else:
                image_channel = tile_image_from_file(path, bands)

            assert image_channel is not None, "Dataset channel {} not retrieved: {}".format(channel["name"], path)

            image = np.concatenate((image, image_channel), axis=2) if image is not None else image_channel

        if self.mode in ["train", "eval"]:
            assert tile == self.tiles["labels"][i][0], "Dataset mask inconsistency"

            mask = tile_label_from_file(self.tiles["labels"][i][1])
            assert mask is not None, "Dataset mask not retrieved"

            weight = self.tiles_weights[tile] if self.tiles_weights is not None and tile in self.tiles_weights else 1.0

            image, mask = to_tensor(
                self.config, self.shape_in[1:3], image, mask=mask, data_augmentation=self.data_augmentation
            )
            return image, mask, tile, weight

        if self.mode in ["predict"]:
            image = to_tensor(self.config, self.shape_in[1:3], image, resize=False, data_augmentation=False)
            return image, torch.IntTensor([tile.x, tile.y, tile.z])
