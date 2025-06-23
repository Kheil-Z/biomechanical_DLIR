"""
SimulatedDataset

Dataset called "rigid" in paper
Synthetic dataset for controlled registration experiments with simple geometric shapes.
Generates 3D volumes featuring a rotating object and corresponding rigidity masks.

Supports both training and validation modes. Each sample includes a fixed and moving image,
along with a binary mask indicating regions for rigidity loss.
"""

import numpy as np
import torch
from monai.data import CacheDataset, Dataset
from src.datasets import BaseDataset


class SimulatedDataset(BaseDataset):
    def __init__(self, params, mode, data_dir, cache_rate=0.0):
        assert mode in [
            "train",
            "val",
        ], f"Unrecognized dataset mode. Got '{mode}', expected 'train' or 'val' "
        self.mode = mode
        self.shape = params["shape"]
        super().__init__(params)
        if cache_rate > 0:
            self.dataset = CacheDataset(
                data=self.data_dict, cache_rate=cache_rate, num_workers=4
            )
        else:
            self.dataset = Dataset(data=self.data_dict)

    def _generate_rotating_image(self):
        im = np.zeros(self.shape)
        mask = np.zeros(self.shape)  # 0 means DetJac loss / 1 means rigidity loss

        # 1) generate the image properties
        center_x = (self.shape[0] // 2) + int(4 * np.random.randn())
        center_y = (self.shape[1] // 2) + int(4 * np.random.randn())
        center_z = (self.shape[2] // 2) + int(4 * np.random.randn())
        rotation = 0.2 * np.random.randn()

        # 2) create the image and set the mask locations for rigidity constraints
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(center_z - 10, center_z + 10):
                    if (i < center_x + 4 + (j - center_y) * np.sin(rotation)) and (
                        i > center_x - 4 + (j - center_y) * np.sin(rotation)
                    ):
                        if (j < center_y + 15 - (i - center_x) * np.sin(rotation)) and (
                            j > center_y - 15 - (i - center_x) * np.sin(rotation)
                        ):
                            im[i, j, k] = 1.0
                            mask[i, j, k] = 1

        # 3) smooth the image
        im[1:-1, 1:-1, 1:-1] = (
            im[1:-1, 1:-1, 1:-1]
            + im[1:-1, 1:-1, 0:-2]
            + im[1:-1, 1:-1, 2:]
            + im[1:-1, 0:-2, 1:-1]
            + im[1:-1, 2:, 1:-1]
            + im[0:-2, 1:-1, 1:-1]
            + im[2:, 1:-1, 1:-1]
        ) / 9.0
        im = torch.tensor(im).float().unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        return im, mask

    def make_data_dict(self):
        data_dict = []
        for i in range(self.params["data_len"][self.mode]):
            fixed_im, loss_mask = self._generate_rotating_image()
            moving_im, _ = self._generate_rotating_image()
            data = {"fixed": fixed_im, "moving": moving_im, "loss_mask": loss_mask}
            data_dict.append(data)
        return data_dict

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def __len__(self):
        return self.params["data_len"][self.mode]
