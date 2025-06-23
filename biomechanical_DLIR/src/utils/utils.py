"""
utils.py

This module provides general-purpose utility functions for reproducibility,
logging, memory profiling, and workflow setup in medical image registration pipelines.

Key Functionalities:
- Logging setup using `loguru` with color-coded outputs for metrics and memory usage.
- Reproducibility utilities including deterministic seeding.
- System memory monitoring for debugging resource usage.

Functions:
- setup_logging: Initializes logger and optionally writes logs to file.
- logger_dict_format: Nicely formats and colorizes metric dictionaries for terminal output.
- seed_all: Sets seeds across PyTorch, NumPy, and Python for reproducibility.
- log_memory_usage: Logs current system memory usage for diagnostics.

Dependencies:
- loguru
- psutil
- monai
- torch
- numpy

Usage:
    from utils import setup_logging, seed_all, log_memory_usage
    setup_logging(params)
    seed_all(42)
    total, used, pct = log_memory_usage()
"""

import os
import random

import numpy as np
import psutil
import torch
from loguru import logger
from monai.utils import set_determinism


# Set up logging
def setup_logging(params, save_log=True):
    if save_log:
        logger.add(
            os.path.join(params["out_folder"], "file.log"),
            colorize=False,
            backtrace=True,
            diagnose=True,
        )
    logger.debug("Program started ... ")
    logger.debug("____" * 10)


def logger_dict_format(d):
    dict_length = len(d)
    keys_per_line = 4 if dict_length % 4 == 0 else 5
    log = "\n \t "
    for key_num, key in enumerate(d, 1):
        log += f" <blue>{key}</blue> : <red>{d[key]:.3f}</red> ; "
        if key_num % keys_per_line == 0:
            log += "\n \t "
    logger.opt(colors=True).info(log)


def seed_all(seed=98):
    set_determinism(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    # warnings.filterwarnings("ignore")


def log_memory_usage():
    total = round(psutil.virtual_memory().total / 1000 / 1000, 4)
    used = round(psutil.virtual_memory().used / 1000 / 1000, 4)
    pct = round(used / total * 100, 1)
    logger.opt(colors=True).debug(
        f"Current memory usage is: <white>{used}</white> / <white>{total}</white> MB (<red>{pct} %</red>)"
    )
    return total, used, pct
