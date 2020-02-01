import unittest

import torch
import mercantile

from neat_eo.loaders.semseg import SemSeg


class TestSemSeg(unittest.TestCase):
    def test_len(self):
        path = "tests/fixtures"
        config = {
            "channels": [{"name": "images", "bands": [1, 2, 3]}],
            "classes": [{"title": "Building", "color": "deeppink"}],
            "model": {"ts": (512, 512)},
            "train": {"pretrained": True, "da": {"name": "RGB", "p": 1.0}},
        }

        # mode train
        dataset = SemSeg(config, (512, 512), path, mode="train")
        self.assertEqual(len(dataset), 3)

        # mode predict
        dataset = SemSeg(config, (512, 512), path, mode="predict")
        self.assertEqual(len(dataset), 3)

    def test_getitem(self):
        path = "tests/fixtures"
        config = {
            "channels": [{"name": "images", "bands": [1, 2, 3]}],
            "classes": [{"title": "Building", "color": "deeppink"}],
            "model": {"ts": (512, 512)},
            "train": {"pretrained": True, "da": {"name": "RGB", "p": 1.0}},
        }

        # mode train
        dataset = SemSeg(config, (512, 512), path, mode="train")
        image, mask, tile, weight = dataset[0]

        assert tile == mercantile.Tile(69105, 105093, 18)
        self.assertEqual(image.shape, torch.Size([3, 512, 512]))

        # mode predict
        dataset = SemSeg(config, (512, 512), path, mode="predict")
        images, tiles = dataset[0]

        self.assertEqual(type(images), torch.Tensor)
