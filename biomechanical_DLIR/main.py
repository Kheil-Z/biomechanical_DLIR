"""
main.py

Configurable training script for deep learning models using PyTorch and Monai.
Supports flexible model/dataset specification via JSON config, metric tracking, WandB logging, 
checkpointing, and optional dataset caching.

Usage:
    python main.py --config path/to/config.json

Some of the expected config keys (note that these are dataset and model specific...):
    - model_name: str
    - model_args: dict
    - dataset_name: str
    - dataset_args: dict
    - loss_name: str
    - loss_args: dict
    - metrics: list[str]
    - optimizer: str
    - optimizer_args: dict
    - lr_scheduler: Optional[str]
    - lr_scheduler_args: dict
    - batch_size: int
    - epochs: int
    - seed: int
    - checkpoint_path: Optional[str]
    - use_wandb: bool
    - project_name: str
    - run_name: str
    - cache_dataset: bool
    - save_best: bool

Example:
    python main.py --config configs/train_config.json
"""

import os
import numpy as np
import argparse
import shutil

from loguru import logger
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torch.optim as optim

import wandb
import hashlib
import copy

from src.models import create_network_class
from src.losses import create_loss_instance
from src.datasets import load_dataset
from src.utils.utils import seed_all, setup_logging, log_memory_usage, logger_dict_format

from collections import defaultdict
import json

def read_json(path):
    """
    Reads a JSON file from the specified path and returns its contents as a Python dictionary.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    with open(path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    return json_object


def update_params_from_json(json_args,params): 
    """
    Updates the given parameters dictionary with values from a JSON dictionary.

    Args:
        json_args (dict): The dictionary containing the JSON arguments.
        params (dict): The dictionary to be updated with the JSON arguments.

    Returns:
        dict: The updated parameters dictionary.
    """
    for key in json_args.keys():
        params[key] = json_args[key]
    return params


def update_params_args(args, params):
    """
    Updates the parameters dictionary with values from the command-line arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        params (dict): The dictionary to be updated with the command-line arguments.

    Returns:
        dict: The updated parameters dictionary.
    """
    for key in args.__dict__.keys():
        if key in params.keys():
            params[key] = getattr(args, key)
    if args.checkpoint is not None :
        params["cpkt"] =  args.checkpoint
    params["data_dir"] = args.data_dir
    params["cache_rate"] = args.cache_rate
    return params

def flatten_data(y):
    """
    Flattens a nested data structure (dict, list, or other types) into a flat dictionary.
    Used to hash json properly and generate unique, reproduciblle name for model if none is given.

    Args:
        y (dict or list): The data structure to flatten.

    Returns:
        dict: The flattened dictionary.
    """
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def make_hash(model_args):
    """
    Generates an MD5 hash string for the given model arguments.
    Used to hash json properly and generate unique, reproduciblle name for model if none is given.

    Args:
        model_args (dict): The dictionary containing the model arguments.

    Returns:
        str: The generated MD5 hash.
    """
    flattened = flatten_data(copy.deepcopy(model_args))
    hash_code = hashlib.md5()
    for k in sorted(flattened):
        hash_code.update(str(k).encode('utf-8'))
        hash_code.update(str(flattened[k]).encode('utf-8'))
    return hash_code.hexdigest()

def make_model(params, device):
    """
    Creates and initializes a model, optimizer, and scheduler based on the provided parameters.

    Args:
        params (dict): The dictionary containing model, optimizer, and scheduler parameters.
        device (torch.device): The device (CPU or GPU) where the model will be trained.

    Returns:
        tuple: A tuple containing the model, optimizer, scheduler, and starting epoch.
    """
    logger.opt(colors=True).info(f'Creating network network of type <red>{params["model_class"]}</red>')
    model_class = create_network_class(params["model_class"])
    model = model_class(params,device)
    # model = model.to(device)
    optimizer = getattr(optim, params["optimizer"])(model.parameters(),
                                                    lr=params["lr"],
                                                    *params["optimizer_kwargs"]) 
    scheduler = getattr(optim.lr_scheduler, params["scheduler"])(optimizer,
                                                                 **params["scheduler_kwargs"]) 
    start_epoch = 0
    # Check if loading from checkpoint
    if params["cpkt"] != "none":
        cpkt = params["cpkt"]
        logger.info(f" Loading model weights from {cpkt}")
        # model.load_state_dict(torch.load(cpkt))
        if os.path.exists(cpkt): 
            checkpoint = torch.load(cpkt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # TODO
            start_epoch = checkpoint['epoch']
        else:
            logger.warning(f"Checkpoint file {cpkt} does not exist.")
            raise FileNotFoundError(f"Checkpoint file {cpkt} does not exist.")
    return model, optimizer, scheduler, start_epoch


def make_data(params,device, mode="train"):
    """
    Creates a data loader for the specified mode (train or validation) and returns it along with the dataset length.

    Args:
        params (dict): The dictionary containing dataset parameters.
        device (torch.device): The device (CPU or GPU) for data processing.
        mode (str, optional): The mode for data loading ("train" or "val"). Defaults to "train".

    Returns:
        tuple: A tuple containing the data loader and the length of the dataset.
    """

    kwargs = {"batch_size":params['batch_size'],
              "shuffle":True,
              "num_workers": 4, # Change to 0 if _pickle.PicklingError: Can't pickle
              "drop_last": True}
    if mode == "val":
        kwargs["batch_size"] = 1
        kwargs["shuffle"] = False
        kwargs["num_workers"] = 0
        kwargs["drop_last"] = False

    dataset_class = load_dataset(params["dataset"])
    dataset = dataset_class(params, mode,params["data_dir"], cache_rate=params["cache_rate"])
    loader = torch.utils.data.DataLoader(
                            dataset,
                            pin_memory=True,
                            **kwargs)
    logger.opt(colors=True).info(f' \t created {mode} data: length:<blue>{len(dataset)}</blue>')
    log_memory_usage()
    return loader,len(dataset)

def init_loss_funcs(params):
    """
    Initializes the loss functions based on the provided parameters.

    Args:
        params (dict): The dictionary containing loss function parameters.

    Returns:
        list: A list of tuples containing the loss function name, weight, and function instance.
    """
    loss_params = params['loss']
    loss_funcs = []
    log = " \t "
    for key,weight in loss_params.items():
        loss_func = create_loss_instance(key,params)
        loss_funcs.append((key,weight,loss_func))
        log += f" <blue>{weight}</blue>*<red>{key}</red> +"
    logger.opt(colors=True).info(log[:-1])
    return loss_funcs

def compute_loss(inputs,ddf,loss_funcs): 
    """
    Computes the loss for a batch of inputs based on the provided loss functions.

    Args:
        inputs (dict): The inputs for the model.
        ddf (dict): The model outputs.
        loss_funcs (list): A list of tuples containing the loss function name, weight, and function.

    Returns:
        tuple: A tuple containing a dictionary of individual losses and the total aggregated loss.
    """
    sample_loss = {}
    total_loss = 0.
    for (key,weight,loss_func) in loss_funcs:
        curr_loss = loss_func.loss(inputs,ddf)
        # Append loss value and aggregate total loss:
        sample_loss[key] = curr_loss.item()
        total_loss += curr_loss*weight # TODO weight here?
    return sample_loss, total_loss

def compute_metrics(inputs,ddf,metric_funcs,metric_values,prefix="",suffix=""): 
    """
    Computes the evaluation metrics for a batch of inputs and appends them to the provided metric values.

    Args:
        inputs (dict): The inputs for the model.
        ddf (dict): The model outputs.
        metric_funcs (list): A list of metric functions.
        metric_values (dict): A dictionary of current metric values to update.
        prefix (str, optional): The prefix for the metric names. Defaults to an empty string.
        suffix (str, optional): The suffix for the metric names. Defaults to an empty string.

    Returns:
        dict: The updated metric values.
    """
    for (_,_,loss_func) in metric_funcs:
        curr_metrics = loss_func.metric(inputs,ddf)
        # Some loss functions return several metrics (e.g DetJac returns num negative and num positive dets...)
        for metric_key in curr_metrics :
            metric_values[prefix+metric_key+suffix].append(curr_metrics[metric_key]*ddf.shape[0]) # TODO since was averaged over batch...
    return metric_values

def train_epoch(model, DL, optimizer, scheduler, loss_funcs,metric_funcs, mode="train"):
    """
    Executes a single training or validation epoch, computing the losses and metrics for each batch.

    Args:
        model (torch.nn.Module): The deep learning model.
        DL (torch.utils.data.DataLoader): The data loader for the current mode (train or val).
        optimizer (torch.optim.Optimizer): The optimizer for backpropagation.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        loss_funcs (list): A list of loss functions.
        metric_funcs (list): A list of metric functions.
        mode (str, optional): The mode for the epoch ("train" or "val"). Defaults to "train".

    Returns:
        tuple: A tuple containing the epoch losses and metrics.
    """
    if mode=="train" : 
        model.train()
    else:
        model.eval()
    epoch_loss = defaultdict(list)
    total_loss = 0.
    total_metric_values = defaultdict(list)
    for batch in DL:
        # get predictions dict
        ddf = model(batch) 
        # total_loss = 0
        sample_loss, loss = compute_loss(batch,ddf,loss_funcs) # TODO
        for key in sample_loss.keys():
            # Append to epoch loss list
            epoch_loss[mode+"_"+key].append(sample_loss[key]*ddf.shape[0]) 
        # If training, backprop loss
        if mode=="train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # Get metrics : 
            total_metric_values = compute_metrics(batch,ddf,metric_funcs,total_metric_values)
        total_loss += loss.item()

    if mode=="val":
        scheduler.step(total_loss) # TODO ?
        # and average metrics:
        for key in total_metric_values:
            # total_metric_values[key] = np.mean(total_metric_values[key])
            total_metric_values[key] = (np.sum(total_metric_values[key])/len(DL.dataset)) 
        # total_metric_values["curr_lr"] = optimizer.param_groups[0]['lr']
    # epoch_loss[mode+"_"+key]
    # Average loss:
    for key in epoch_loss.keys():
        # epoch_loss[key] = (np.mean(epoch_loss[key])) 
        epoch_loss[key] = (np.sum(epoch_loss[key])/len(DL.dataset)) 
    logger.info("Loss values:")
    logger_dict_format(epoch_loss)
    if mode=="val":
        logger.info("Metric values:")
        logger_dict_format(total_metric_values)
    return epoch_loss, total_metric_values



def init_loss_funcs(params):#loss_params):
    loss_params = params['loss']
    loss_funcs = []
    log = " \t "
    for key,weight in loss_params.items():
        loss_func = create_loss_instance(key,params)
        loss_funcs.append((key,weight,loss_func))
        log += f" <blue>{weight}</blue>*<red>{key}</red> +"
    logger.opt(colors=True).info(log[:-1])
    return loss_funcs

def train_val_reg(model,trainDL,valDL,params, optimizer, scheduler):
    """
    Runs the full training and validation process, including saving the best model based on validation loss.

    Args:
        model (torch.nn.Module): The deep learning model.
        trainDL (torch.utils.data.DataLoader): The training data loader.
        valDL (torch.utils.data.DataLoader): The validation data loader.
        params (dict): The dictionary containing training parameters.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.

    Returns:
        tuple: A tuple containing the training and validation loss history and the trained model.
    """
    train_loss_history = defaultdict(list)
    val_loss_history = defaultdict(list)

    logger.info("----- Initialising loss functions -----")
    loss_funcs = init_loss_funcs(params)
    
    logger.info("----- Initialising Metric functions -----")
    # TODO change... what if Dice? what if... ? TODO TODO 
    basic_metrics = {"MSE":1.0,"NCC":1.0,"DetJac":1.0}
    if params["dataset"].lower() == "simulated":
        basic_metrics["Rigidity"] = 1.0
    if params["dataset"].lower() == "simulatedshear":
        basic_metrics["RigidityDetShearing"] = 1.0
    if params["dataset"].lower() in ["abdomenctct","abdomenctctroi","nlst"]:
        basic_metrics["Dice"] = 1.0
    
    copy_params = copy.deepcopy(params)
    copy_params["loss"] = basic_metrics
    metric_funcs = init_loss_funcs(copy_params)#{"loss":basic_metrics})
    best_val_sim = np.inf
    # Loop over epochs
    for epoch in range(params["start_epoch"], params["epochs"]):
        logger.opt(colors=True).info(f"Training loop <blue>{epoch}</blue>/<red>{params['epochs']}</red>")
        # Train loop
        losses_train,_ = train_epoch(model, trainDL, optimizer, scheduler,loss_funcs,metric_funcs, mode="train")
        # Validation loop
        with torch.no_grad():
            losses_val,val_metric_values = train_epoch(model,valDL, None, scheduler, loss_funcs,metric_funcs, mode="val")

        for key in losses_train.keys():
            train_loss_history[key].append(losses_train[key])
        for key in losses_val.keys():
            val_loss_history[key].append(losses_val[key])
        
        # if wandb ...
        if params["wandb"]:
            wandb.log(losses_train | losses_val | val_metric_values)
        
        # Check if should save model for best val loss
        val_sim = losses_val[f"val_{list(params['loss'].keys())[0]}"]
        if val_sim<=best_val_sim:
            best_val_sim = val_sim
            save_model_name = 'best_val_similarity.pth'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, os.path.join(params["checkpoints_paths"], save_model_name))
            logger.opt(colors=True).info(f"Model saved for best Validation Similarity at epoch <blue>{epoch}</blue> for: <red>{best_val_sim}</red>")
        
        # Also save most recent and every 50 epochs
        save_model_name = str(epoch) + '-net.pth' if (epoch>0 and (epoch % 50 == 0)) else 'most_recent.pth'
        # save_model_name = 'most_recent.pth'
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(params["checkpoints_paths"], save_model_name))

    return train_loss_history, val_loss_history, model


def train(model, params, trainDL, valDL, optimizer, scheduler):
    """
    Trains the model for the specified number of epochs, logging the losses and metrics, and saving the results.

    Args:
        model (torch.nn.Module): The deep learning model.
        params (dict): The dictionary containing training parameters.
        trainDL (torch.utils.data.DataLoader): The training data loader.
        valDL (torch.utils.data.DataLoader): The validation data loader.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.

    Returns:
        torch.nn.Module: The trained model.
    """
    logger.info(f'Network train starting... Outputs will be saved to {params["out_folder"]}')
    # Create save paths if not exist:
    params["checkpoints_paths"] = os.path.join(params["out_folder"],"checkpoints/")
    if not os.path.exists(params["checkpoints_paths"]):
        os.makedirs(params["checkpoints_paths"])
    
    
    ################ Train Loop ################
    train_loss_history, val_loss_history, model = train_val_reg(model,trainDL,valDL,
                                                    params, optimizer, scheduler)

    ################ Save Loss plots ################
    keys_train = list(train_loss_history.keys())
    keys_val = list(val_loss_history.keys())
    title_list = keys_train
    fig,axes = plt.subplots(ncols=len(title_list), figsize=(20,8))

    for i,ax in enumerate(axes):
        ax.plot(np.array(train_loss_history[keys_train[i]]), label="Train")
        if len(val_loss_history) > 0:
            ax.plot(val_loss_history[keys_val[i]], label="Val")
        ax.set_title(title_list[i])
        ax.legend()
    fig.tight_layout()
    # plt.savefig(f"{save_name}_LOSSES{name_suffix}.png")
    plt.savefig(os.path.join(params["out_folder"],f"train_val_losses.png"))
    plt.show()

    # del trainDL
    # del train_loader
    # del val_dataset
    # del val_loader
    # gc.collect()
    return model

def make_wandb_config_readable(params_dict):
    """
    Converts the parameters dictionary into a format suitable for logging with WandB.

    Args:
        params_dict (dict): The dictionary containing model parameters.

    Returns:
        dict: The formatted WandB configuration.
    """
    wandb_config = {}
    simple_keys = ["model_class","model_kwargs","integration_steps","cpkt",
                   "batch_size","epochs",
                   "lr",
                   "optimizer","optimizer_kwargs",
                   "scheduler","scheduler_kwargs",
                   "dataset","shape","cache_rate","dice_labels"]
    for k in simple_keys:
        if k in params_dict :
            wandb_config[k] = params_dict[k]
    # For the rest of the params... for easier filtering on wandb, we restructure them:
    if "data_len" in params_dict:
        wandb_config["train_len"] = params_dict["data_len"]["train"]
        wandb_config["val_len"] = params_dict["data_len"]["val"]
    # "Loss" key will always be there, for easier filtering on WandB, expand all possible losses:
    all_loss_keys = ["Bending","DetJac","Dice","DiceCE","MI","MSE","NCC","Shearing","StrainDet","StrainTensor","Shearing","StrainDetShearing"]
    for key in all_loss_keys:
        wandb_config[key] = 0
    used_loss_keys = list(params_dict["loss"].keys())
    for key in used_loss_keys:
        wandb_config[key] = params_dict["loss"][key]
    return wandb_config

def main(args):
    """
    The main entry point for the training script. Parses arguments, initializes configurations, 
    sets up the training environment, and runs the training process.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    # Set random seeds
    seed_all(seed=args.seed)

    # get Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read model and data parametrisation
    model_args = read_json(args.model_config)
    
    # Make subsequent parameters object which contains required training and validation info
    params = {}
    params = update_params_from_json(model_args,params)
    # Possibility to override some parameters:
    params = update_params_args(args, params)

    config_file_name = args.model_config.split(os.sep)[-1].split(".json")[0]
    if args.log_wandb:
        wandb_config = make_wandb_config_readable(params)
        params["wandb"] = True
        if "run_name" not in params:
            run = wandb.init(project=f"{params['dataset']}",config=wandb_config, tags=config_file_name)
            params["run_name"] = config_file_name + "_" + run.name
            # Update wandb
            run.name = params["run_name"]
            run.save()
        else:
            run = wandb.init(project=f"{params['dataset']}",name=params["run_name"],config=wandb_config)
    elif "run_name" not in params:
        params["run_name"] = config_file_name + "_" + make_hash(model_args)
        params["wandb"] = False
    params["out_folder"] = os.path.join(params["out_folder"],params["dataset"],params["run_name"])
    
    if not os.path.exists(params["out_folder"]):
        os.makedirs(params["out_folder"])
        
    # Make sure arguments filed is copied to results dir for later..
    shutil.copy(args.model_config, os.path.join(params["out_folder"],"model_config.json"))
    # # Setup Logger :
    setup_logging(params)

    # Log used params:
    logger.info('\t Model Information')
    for item in model_args:
        logger.opt(colors=True).info(f' \t {item}:<blue>{model_args[item]}</blue>')

    # Make Datasets:
    logger.debug('\t Creating Datasets')
    log_memory_usage()
    trainDL,train_num_pairs = make_data(params, device, mode="train")
    valDL,val_num_pairs = make_data(params, device, mode="val")
    params["train_num_pairs"] = train_num_pairs
    params["val_num_pairs"] = val_num_pairs

    # Make model
    model, optimizer, scheduler, start_epoch = make_model(params, device)
    params["start_epoch"] = start_epoch

    model = train(model, params, trainDL, valDL, optimizer, scheduler)

    if args.log_wandb:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', '-m', type=os.path.abspath, help='path to json configuration file for model', required=True)
    parser.add_argument('--cache_rate', '-c',type=float, default=1.0)
    
    parser.add_argument('--log_wandb', action=argparse.BooleanOptionalAction, help="Use flag to log metrics to WandB, leave empty or '--no-log_wandb' not to.")
    parser.add_argument('--seed', default=98)
    parser.add_argument('--data_dir', type=os.path.abspath, default="Datasets/", help='Path to data locations',)
    # parser.add_argument('--out_folder','-o', type=os.path.abspath, default="results/", help='Path to save outputs',)

    # Override params from json...:
    parser.add_argument("--checkpoint", default=None, type=os.path.abspath, help="Path to pretrained models. During training, continue from this checkpoint. During evaluation, evaluate this checkpoint.")
    parser.add_argument('--out_folder','-o', type=os.path.abspath, default="results/", help='Path to save outputs',)
    args = parser.parse_args()
    
    main(args)


# python main.py --model_config configs/simulated_configs/simu_strainDet.json 