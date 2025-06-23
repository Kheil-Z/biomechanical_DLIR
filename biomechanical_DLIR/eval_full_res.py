"""
This script performs evaluation of a pretrained model on a specified dataset.
The model is evaluated by computing various metrics and optionally saving sample outputs during the evaluation.

The code supports up-sampling images back to their original size for evaluation,
 providing the ability to compute metrics like MSE, NCC, DetJac, and others.
   The evaluation results can be stored in a user-specified output folder, and sample outputs are saved if required.

The following steps are performed:
1. Load model and dataset configuration from a checkpoint.
2. Set up necessary parameters and logging.
3. Load the dataset and create data loaders for evaluation.
4. Initialize the loss functions and metric functions.
5. Evaluate the model on the dataset and compute metrics.
6. Optionally save sample outputs during evaluation, including fixed, moving, and transformed images along with displacement fields.

Command-line arguments:
    --checkpoint: Path to the pretrained model for evaluation.
    --cache_rate: Cache rate for the dataset.
    --seed: Random seed for reproducibility.
    --data_dir: Path to the dataset directory.
    --save_number: Number of samples to save during evaluation.
    --mode: Mode for evaluation (e.g., 'val').
    --out_folder: Path to save evaluation results and outputs.

Dependencies:
    - PyTorch
    - MONAI
    - Nibabel
    - Matplotlib
    - Pandas
    - WandB (optional)
    - Loguru (for logging)
"""

import argparse
import copy
import hashlib
import json
import os
import time
from collections import defaultdict

import monai.transforms as MTransforms
import nibabel as nib
import numpy as np
import torch
import torch.optim as optim

# AbdomenCTCT
from loguru import logger
from monai.networks.blocks import Warp
from src.datasets import load_dataset
from src.losses import create_loss_instance
from src.models import create_network_class
from src.utils.utils import (
    log_memory_usage,
    logger_dict_format,
    seed_all,
    setup_logging,
)


def read_json(path):
    with open(path, "r") as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    return json_object


def update_params_from_json(json_args, params):
    for key in json_args.keys():
        params[key] = json_args[key]
    return params


def update_params_args(args, params):
    for key in args.__dict__.keys():
        if key in params.keys():
            params[key] = getattr(args, key)
    if args.checkpoint is not None:
        params["cpkt"] = args.checkpoint
    params["data_dir"] = args.data_dir
    params["cache_rate"] = args.cache_rate
    return params


def flatten_data(y):
    out = {}

    def flatten(x, name=""):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + "_")
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def make_hash(model_args):
    flattened = flatten_data(copy.deepcopy(model_args))
    hash_code = hashlib.md5()
    for k in sorted(flattened):
        hash_code.update(str(k).encode("utf-8"))
        hash_code.update(str(flattened[k]).encode("utf-8"))
    return hash_code.hexdigest()


def make_model(params, device):
    logger.opt(colors=True).info(
        f'Creating network network of type <red>{params["model_class"]}</red>'
    )
    model_class = create_network_class(params["model_class"])
    model = model_class(params, device)
    # model = model.to(device)
    optimizer = getattr(optim, params["optimizer"])(
        model.parameters(), lr=params["lr"], *params["optimizer_kwargs"]
    )
    scheduler = getattr(optim.lr_scheduler, params["scheduler"])(
        optimizer, **params["scheduler_kwargs"]
    )
    start_epoch = 0
    # Check if loading from checkpoint
    if params["cpkt"] != "none":
        cpkt = params["cpkt"]
        logger.info(f" Loading model weights from {cpkt}")
        # model.load_state_dict(torch.load(cpkt))
        if os.path.exists(cpkt):
            checkpoint = torch.load(cpkt, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # TODO
            start_epoch = checkpoint["epoch"]
        else:
            logger.warning(f"Checkpoint file {cpkt} does not exist.")
            raise FileNotFoundError(f"Checkpoint file {cpkt} does not exist.")
    return model, optimizer, scheduler, start_epoch


# TODO ARGS : Done this, needs : params['batch_size'], params["dataset"] <- "NLST"m , **params["dataset_args"]
def make_data(params, device, mode="train"):
    kwargs = {"batch_size": 1, "shuffle": False, "num_workers": 0, "drop_last": True}
    # if mode == "val":
    #     kwargs["batch_size"] = 1
    #     kwargs["shuffle"] = False
    #     kwargs["num_workers"] = 0
    #     kwargs["drop_last"] = False

    dataset_class = load_dataset(params["dataset"])
    dataset = dataset_class(
        params, mode, params["data_dir"], cache_rate=params["cache_rate"]
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        # collate_fn=lambda x: tuple(val.to(device) for (key,val) in list_data_collate(x).items())
        # collate_fn=lambda x: [val.to(device) for (key,val) in default_collate(x).items()],
        # collate_fn=lambda x: tuple(x_.to(device) for x_ in list_data_collate(x))
        **kwargs,
    )
    logger.opt(colors=True).info(
        f" \t created {mode} data: length:<blue>{len(dataset)}</blue>"
    )
    log_memory_usage()
    return loader, len(dataset), dataset


def init_loss_funcs(params):  # loss_params):
    loss_params = params["loss"]
    loss_funcs = []
    log = " \t "
    for key, weight in loss_params.items():
        loss_func = create_loss_instance(key, params)
        loss_funcs.append((key, weight, loss_func))
        log += f" <blue>{weight}</blue>*<red>{key}</red> +"
    logger.opt(colors=True).info(log[:-1])
    return loss_funcs


def compute_loss(inputs, ddf, loss_funcs):
    sample_loss = {}
    total_loss = 0.0
    for key, weight, loss_func in loss_funcs:
        curr_loss = loss_func.loss(inputs, ddf)
        # Append loss value and aggregate total loss:
        sample_loss[key] = curr_loss.item()
        total_loss += curr_loss * weight  # TODO weight here?
    return sample_loss, total_loss


def compute_metrics(inputs, ddf, metric_funcs, metric_values, prefix="", suffix=""):
    for _, _, loss_func in metric_funcs:
        curr_metrics = loss_func.metric(inputs, ddf)
        # Some loss functions return several metrics (e.g DetJac returns num negative and num positive dets...)
        for metric_key in curr_metrics:
            metric_values[prefix + metric_key + suffix].append(curr_metrics[metric_key])
    return metric_values


def save_sample(params, data, ddf, name):
    #
    if not os.path.exists(os.path.join(params["out_folder"], name)):
        os.makedirs(os.path.join(params["out_folder"], name))

    device = ddf.get_device()
    fixed = data["fixed"].to(device)
    moving = data["moving"].to(device)
    moved = Warp()(moving, ddf)

    # Get or make affine for nibabel :
    if (hasattr(fixed, "meta")) and ("afffine" in fixed.meta):
        affine = fixed.meta["affine"]
    else:
        affine = np.eye(4)

    nib.save(
        nib.Nifti1Image(ddf.squeeze().cpu().permute(1, 2, 3, 0).numpy(), affine),
        os.path.join(params["out_folder"], name, "ddf.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(moved.squeeze().cpu().numpy(), affine),
        os.path.join(params["out_folder"], name, "moved.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(fixed.squeeze().cpu().numpy(), affine),
        os.path.join(params["out_folder"], name, "fixed.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(moving.squeeze().cpu().numpy(), affine),
        os.path.join(params["out_folder"], name, "moving.nii.gz"),
    )

    if "loss_mask" in data:
        loss_mask = data["loss_mask"].to(device)
        nib.save(
            nib.Nifti1Image(loss_mask.squeeze().cpu().numpy(), affine),
            os.path.join(params["out_folder"], name, "loss_mask.nii.gz"),
        )


def save_sample_full_size(params, data, ddf, name):
    """
    Saves the input data, predicted displacement field, and transformed images back to their original size.

    Args:
        params (dict): Dictionary of output parameters.
        data (dict): Dictionary containing input data including moving and fixed images.
        ddf (torch.Tensor): The predicted displacement field.
        name (str): The name of the sample to save.
    """
    if not os.path.exists(os.path.join(params["out_folder"], name)):
        os.makedirs(os.path.join(params["out_folder"], name))

    device = ddf.get_device()
    fixed = data["fixed"].to(device)
    moving = data["moving"].to(device)
    if ddf.shape[2::] != moving.shape:
        _, _, z, h, w = moving.shape
        transform_resize = MTransforms.Resize(
            (z, h, w), mode="trilinear", anti_aliasing=True, align_corners=True
        )
        # Create appropriate shapes:
        ddf = ddf.squeeze()
        ddf = transform_resize(ddf.squeeze()).unsqueeze(0)

    moved = Warp()(moving, ddf)

    # Get or make affine for nibabel :
    if (hasattr(fixed, "meta")) and ("afffine" in fixed.meta):
        affine = fixed.meta["affine"]
    else:
        affine = np.eye(4)
    if not os.path.exists(os.path.join(params["out_folder"], name, "full_size/")):
        os.makedirs(os.path.join(params["out_folder"], name, "full_size/"))
    nib.save(
        nib.Nifti1Image(ddf.squeeze().cpu().permute(1, 2, 3, 0).numpy(), affine),
        os.path.join(params["out_folder"], name, "full_size/ddf.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(moved.squeeze().cpu().numpy(), affine),
        os.path.join(params["out_folder"], name, "full_size/moved.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(fixed.squeeze().cpu().numpy(), affine),
        os.path.join(params["out_folder"], name, "full_size/fixed.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(moving.squeeze().cpu().numpy(), affine),
        os.path.join(params["out_folder"], name, "full_size/moving.nii.gz"),
    )

    if "loss_mask" in data:
        loss_mask = data["loss_mask"].to(device)
        nib.save(
            nib.Nifti1Image(loss_mask.squeeze().cpu().numpy(), affine),
            os.path.join(params["out_folder"], name, "full_size/loss_mask.nii.gz"),
        )


def main(args):
    # Set random seeds
    seed_all(seed=args.seed)

    # get Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get config path :
    model_config_path = os.path.join(
        *args.checkpoint.split(os.sep)[:-2], "model_config.json"
    )

    # Read model and data parametrisation
    model_args = read_json(model_config_path)

    # Make subsequent parameters object which contains required training and validation info
    params = {}
    params = update_params_from_json(model_args, params)
    # Possibility to override some parameters:
    params = update_params_args(args, params)

    # Loger:
    setup_logging(params, save_log=False)
    # Setup Save path:
    if args.out_folder:
        params["out_folder"] = args.out_folder
    else:
        params["out_folder"] = os.path.join(
            *args.checkpoint.split(os.sep)[:-2]
        ).replace(params["dataset"], params["dataset"] + "/eval")
        params["out_folder"] += "_" + args.checkpoint.split(os.sep)[-1].replace(
            ".pth", ""
        )
    if args.mode != "val":
        params["out_folder"] += f"mode={args.mode}"
    if not os.path.exists(params["out_folder"]):
        os.makedirs(params["out_folder"])

    # If was trained using additional labels, pop them to make sure all labels are used for evaluation..
    params.pop("additional_labels", None)
    params.pop("map_labels", None)
    # Make dataset
    valDL_normal, _, dataset_normal = make_data(params, device, mode=args.mode)
    # Make full size dataset:
    small_shape = params["shape"]
    params.pop("shape", None)
    valDL_full_size, _, dataset_full_size = make_data(params, device, mode=args.mode)
    # dataset = torch.utils.data.TensorDataset(dataset_normal, dataset_full_size)
    # dataloader_both = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    params["shape"] = small_shape

    # # Make loss func :
    # loss_funcs_val = init_loss_funcs(params)#params['loss']["val"])
    # # TODO change... what if Dice? what if... ? TODO TODO

    basic_metrics = {"MSE": 1.0, "NCC": 1.0, "DetJac": 1.0}
    before_registration_metrics = {"MSE": 1.0, "NCC": 1.0}
    if params["dataset"].lower() == "simulated":
        basic_metrics["Rigidity"] = 1.0
    if params["dataset"].lower() == "simulatedshear":
        basic_metrics["RigidityDetShearing"] = 1.0
    if params["dataset"].lower() in ["abdomenctct", "abdomenctctroi", "nlst"]:
        basic_metrics["Dice"] = 1.0
        basic_metrics["RigidityDetShearing"] = 1.0
        before_registration_metrics["Dice"] = 1.0
        params["additional_labels"] = dataset_normal.all_label_values

    params["loss"] = basic_metrics
    metric_funcs = init_loss_funcs(params)  # {"loss":basic_metrics})
    params["loss"] = before_registration_metrics
    metric_funcs_no_registration = init_loss_funcs(params)
    # Load model
    model, _, _, _ = make_model(params, device)
    model.eval()
    metric_values = defaultdict(list)
    sample_metrics = {}
    with torch.no_grad():
        # for i,data in enumerate(valDL):
        for i, (data, data_full_size) in enumerate(zip(valDL_normal, valDL_full_size)):

            start_time = time.time()
            print(data["moving"].shape)
            ddf = model(data)
            end_time = time.time()

            # Compute metrics:
            metric_values = compute_metrics(data, ddf, metric_funcs, metric_values)
            metric_values = compute_metrics(
                data,
                torch.zeros_like(ddf),
                metric_funcs_no_registration,
                metric_values,
                prefix="before_",
            )
            metric_values["time"].append(end_time - start_time)
            for key in metric_values:
                sample_metrics[key] = metric_values[key][
                    -1
                ]  # np.mean(metric_values[key])
            logger_dict_format(sample_metrics)

            # If saving samples, save  number of samples and stop evaluating after. If not, evaluate all. :
            if i < args.save_number:
                save_sample(params, data, ddf, f"sample{i}")
                save_sample_full_size(params, data_full_size, ddf, f"sample{i}")
            elif args.save_number != 0:
                break
        # pd.DataFrame(metric_values).to_csv(os.path.join(params["out_folder"],"metrics.csv"),index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pretrained models. During training, continue from this checkpoint. During evaluation, evaluate this checkpoint.",
    )

    parser.add_argument("--cache_rate", "-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument(
        "--data_dir",
        type=os.path.abspath,
        default="Datasets/",
        help="Path to data locations",
    )

    parser.add_argument("--save_number", type=int, default=0)
    parser.add_argument("--mode", default="val")  # or mode="train"

    parser.add_argument(
        "--out_folder", "-o", default=None, help="Path to save outputs or None"
    )

    args = parser.parse_args()
    print(args.out_folder)

    main(args)
