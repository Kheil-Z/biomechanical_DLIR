"""
SimulatedShearDataset

similar to dataset called "Shearing" in paper, but in this version cuboids stay on same line for every pair.
Synthetic dataset designed for studying deformation types in image registration, 
including rigid and shearing transformations.

Each 3D sample includes a fixed and moving volume, a multi-class loss mask 
(indicating regions for different loss terms), and a shear direction vector.
"""

from src.datasets import BaseDataset
from monai.data import Dataset, CacheDataset


import numpy as np
import torch

class SimulatedShearDataset(BaseDataset):
    def __init__(self, params, mode, data_dir, cache_rate=0.0):
        assert mode in ["train", "val"], f"Unrecognized dataset mode. Got '{mode}', expected 'train' or 'val' "
        self.mode = mode
        self.shape = params["shape"] # = [64,64,64]
        super().__init__(params)
        if cache_rate>0:
            self.dataset = CacheDataset(data=self.data_dict, cache_rate=cache_rate, num_workers=4)
        else:
            self.dataset = Dataset(data=self.data_dict)

    def _generate_rotating_image(self,centers=None):
        im=np.zeros([64,64,64])
        mask=np.zeros([64,64,64])  #0 means DetJac loss / 1 means rigidity loss / 2 means shearing loss

        #1) generate the image properties
        if centers:
            center_x,center_y,center_z = centers
        else:
            center_x = (self.shape[0]//2) + int(4*np.random.randn())
            center_y = (self.shape[1]//2) + int(4*np.random.randn())
            center_z = (self.shape[2]//2) + int(4*np.random.randn())
        shift=int(3*np.random.randn())

        #2) create the image and set the mask locations for rigidity constraints
        for i in range(center_x-10-shift,center_x+10-shift):
            for j in range(center_y-10,center_y):
                for k in range(center_z-10,center_z+10):
                    im[i,j,k]=0.8
                    mask[i,j,k]=1
        for i in range(center_x-10+shift,center_x+10+shift):
            for j in range(center_y,center_y+10):
                for k in range(center_z-10,center_z+10):
                    im[i,j,k]=1.0
                    mask[i,j,k]=1

        #3) set the mask locations for shearing constraints
        for i in range(center_x-10-np.abs(shift)-1,center_x+10+np.abs(shift)+1):
            for j in range(center_y-1,center_y+1):
                for k in range(center_z-10,center_z+10):
                    mask[i,j,k]=2
        # TODO : Add epaisseur a lendroit de cissaillement for loss mask
        proj_vector=torch.tensor([0.,1.,0.])

        im = torch.tensor(im).float().unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        return im,mask,proj_vector, (center_x,center_y,center_z)

    def make_data_dict(self):
        data_dict = []
        for i in range(self.params["data_len"][self.mode]):
            fixed_im,_,_,centers = self._generate_rotating_image()
            moving_im, loss_mask, proj_vector,_ = self._generate_rotating_image(centers)
            data = {
                    "fixed": fixed_im,
                    "moving": moving_im,
                    "loss_mask": loss_mask,
                    "proj_vector":proj_vector,
                }
            data_dict.append(data)
        return data_dict
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def __len__(self):
        return self.params["data_len"][self.mode]
    