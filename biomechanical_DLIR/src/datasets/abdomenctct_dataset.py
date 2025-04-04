"""
AbdomenCTCTDataset with our generated annotations

Dataset class for the AbdomenCTCT registration task, built on top of BaseDataset and MONAI's Dataset classes.
Supports both training and validation modes with custom image pairing logic from the challenge-specific JSON file.

Applies intensity normalization, masking, resizing, and optional label filtering/mapping via MONAI transforms.
Returns paired (moving, fixed) CT images with associated labels and masks for registration tasks.
"""

from src.datasets import BaseDataset
from monai.data import Dataset, CacheDataset

import torch
import monai.transforms as MTransforms
import pandas as pd
import os
import json
from itertools import combinations


class AbdomenCTCTDataset(BaseDataset):
    def __init__(self, params, mode,data_dir, cache_rate=1.0):
        assert mode in ["train", "val"], f"Unrecognized dataset mode. Got '{mode}', expected 'train' or 'val' "
        self.mode = mode
        self.data_dir = os.path.join(data_dir,f"AbdomenCTCT_reoriented/")
        self.label_keys = ["label","total","lung_vessels","lung_nodules"] # All additional labels keys, will be filtered out latrer
        self.all_label_values = {"label":list(range(1,14)),
                                 "total":list(range(1,118)),
                                 "lung_vessels":[1],
                                 "lung_nodules":[1]}
        super().__init__(params)
        if cache_rate>0:
            self.dataset = CacheDataset(data=self.data_dict, transform=self.transforms, cache_rate=cache_rate, num_workers=4)
        else:
            self.dataset = Dataset(data=self.data_dict, transform=self.transforms)

    
    def make_data_dict(self):
        """
        Get L2R train/val files from AbdomenMRCT challenge.
        NB: Specificity here lies in that training pairs are not pre-constructed, 
            and some training images are used in the validation set...
        """
        data_json = os.path.join(self.data_dir, "AbdomenCTCT_dataset.json")

        # Open json file containing dataset information
        with open(data_json) as file:
            data = json.load(file)
            
        # Get all pairs of images in validation set:
        val_images = []
        val_pairs = []
        for pair in data["registration_val"]:
            f = pair["fixed"]
            m = pair["moving"]
            if f not in val_images:
                val_images.append(f)
            if m not in val_images:
                val_images.append(m)
            val_pairs.append((f,m))

        # then get images which are not used in the validation set...
        training_images = []
        for pair in data["training"]:
            p = pair["image"]
            if (p not in val_images) and (p not in training_images):
                training_images.append(p)
        files = []
        if self.mode == "train":
            image_list = training_images
            # Construct dict of pairs of images 
            pairs = list(combinations(training_images, 2))
        # if in validation mode, much easier: just get given paris from JSON
        elif self.mode == "val":
            image_list = val_images
            pairs = val_pairs

        self.path_to_idx = {image_list[idx]:idx for idx in range(len(image_list))}
        self.pairs = pairs
        
        # cosntruct Final dictionaries of training and validation (moving,fixed) file paths.
        files = []
        for path in image_list:
            name = os.path.basename(path).split(".")[0]
            files.append(
                {
                    "image": os.path.join(self.data_dir, "imagesTr", name + ".nii.gz"),
                    "label": os.path.join(self.data_dir, "labelsTr", name + ".nii.gz"),
                    "total": os.path.join(self.data_dir, "segmentationsTr/total/", name + ".nii.gz"),

                    #  TODO : others?
                    "lung_vessels": os.path.join(self.data_dir, "segmentationsTr/lung_vessels/", name + ".nii.gz"),
                    "lung_nodules": os.path.join(self.data_dir, "segmentationsTr/lung_nodules/", name + ".nii.gz"),
                    "liver_vessels": os.path.join(self.data_dir, "segmentationsTr/liver_vessels/", name + ".nii.gz"),
                    "body": os.path.join(self.data_dir, "segmentationsTr/body/", name + ".nii.gz"),
                    "strain_mask": os.path.join(self.data_dir, "segmentationsTr/strain_mask/", name + ".nii.gz"),
                    "projection_vectors": os.path.join(self.data_dir, "segmentationsTr/projection_vectors/", name + ".nii.gz"),
                }
            )
        return files
    
    
    def init_transforms(self):
        # Get image shape for network.
        if "shape" in self.params:
            target_res = self.params["shape"]
            spatial_size = target_res 
        else:
            target_res = [192,160,256] 
            spatial_size = [-1,-1,-1,]  # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model

        # Cosntruct monai loader arguments:
        keys = ["image","strain_mask","projection_vectors"]
        align_corners=[True,False,False]
        interpolate_mode=["trilinear","nearest","nearest"]
        anti_aliasing = [True,False,False]

        for key in self.label_keys:
            keys.append(key)
            align_corners.append(False)
            interpolate_mode.append("nearest")
            anti_aliasing.append(False)
        
        # "body" label is always required to mask images.
        keys.append("body")
        # 1) Load image
        list_transforms = [MTransforms.LoadImaged(keys=keys, ensure_channel_first=True)]
        # 4) Scale intesity and min-max normalize
        list_transforms.append(MTransforms.ScaleIntensityRanged(
                                keys=["image"],
                                a_min=-1200,
                                a_max=400,
                                b_min=0.0,
                                b_max=1.0,
                                clip=True,
                            ))
        # 2) Mask images with body mask
        list_transforms.append(MTransforms.MaskIntensityd(keys=list(filter(lambda key: key != "projection_vectors", keys)),
                                                          mask_key="body"))
        # 3) Delete body mask to free memory
        list_transforms.append(MTransforms.DeleteItemsd(keys=["body"]))
        keys.remove("body")
        # 3) Downsample image if required
        list_transforms.append(MTransforms.Resized(
                                keys=keys,
                                mode=interpolate_mode,
                                anti_aliasing=anti_aliasing,
                                spatial_size=spatial_size,
                            ))
        
        # Merge certain label values to have smaller memory requirement:
        if "map_labels" in self.params.keys():
            for key in self.params["map_labels"]:
                original_labels = self.params["map_labels"][key]["original_labels"]
                replacement_labels = self.params["map_labels"][key]["replacement_labels"]
                list_transforms.append(MTransforms.MapLabelValued(
                                        keys=[key],
                                        orig_labels=original_labels,
                                        target_labels=replacement_labels,
                                        ))

        # Keep only wanted labels:
        if "additional_labels" in self.params.keys():
            label_keys = []
            for key in self.label_keys:
                if key in self.params["additional_labels"]:
                    labels = self.params["additional_labels"][key]
                    list_transforms.append(MTransforms.LabelFilterd(
                                            keys=[key],
                                            applied_labels=labels,
                                            ))
                    label_keys.append(key)
                else:
                    list_transforms.append(MTransforms.DeleteItemsd(keys=[key]))
                    keys.remove(key)
            self.label_keys = label_keys
        list_transforms.append(MTransforms.CastToTyped(
                keys=self.label_keys,
                allow_missing_keys = True,
                dtype=torch.float))
        # Compose transformations
        loader = MTransforms.Compose(list_transforms)
        return loader

    def __getitem__(self, idx):
        f,m = self.pairs[idx]
        idx_f,idx_m = self.path_to_idx[f],self.path_to_idx[m]
        item_f = self.dataset[idx_f]
        item_m = self.dataset[idx_m]
        data_dict = {"fixed":item_f["image"],
                     "moving":item_m["image"],
                     "loss_mask":item_m["strain_mask"],
                     "proj_vector":item_m["projection_vectors"]
                    }
        for key in self.label_keys :
            data_dict[f"fixed_{key}"] = item_f[key]
            data_dict[f"moving_{key}"] = item_m[key]
        return data_dict

    def __len__(self):
        return len(self.pairs)