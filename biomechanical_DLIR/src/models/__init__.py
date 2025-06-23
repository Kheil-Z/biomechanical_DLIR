"""
Dynamic model loading and base model definition for neural networks.

This module provides utility functions to dynamically load model classes
from files, as well as a base class to be extended by specific model implementations.

Functions:
    create_network_class(network_name):
        Dynamically loads and returns a model class from a file matching the given name.
        The file must be located in the same directory as this script and should define
        a class with the exact same name as the filename (case-sensitive).

Classes:
    BaseModel:
        An abstract base class for neural networks. All models must inherit from this class
        and implement the `compute_loss` method. Designed to store configuration parameters
        and device information, and optionally handle loss initialization.
"""

import importlib.util
import os
from abc import abstractmethod

import torch.nn as nn

# Directory containing network files
models_directory = os.path.dirname(__file__)


def create_network_class(network_name):
    """
    Create a class of the specified network.

    Args:
        network_name (str): Name of the network.

    Returns:
        network_class: Class of the network.

    Raises:
        FileNotFoundError: If the network file is not found.
        AttributeError: If the network class is not found.
    """

    network_file_path = os.path.join(models_directory, f"{network_name.lower()}.py")
    if not os.path.exists(network_file_path):
        raise FileNotFoundError(
            f"Network file '{network_name.lower()}.py' not found in '{models_directory}'."
        )

    network_module_name = f"models.{network_name.lower()}"
    network_module_spec = importlib.util.spec_from_file_location(
        network_module_name, network_file_path
    )
    network_module = importlib.util.module_from_spec(network_module_spec)
    network_module_spec.loader.exec_module(network_module)

    network_class = getattr(network_module, network_name, None)

    if network_class is None:
        raise AttributeError(
            f"Network class '{network_name}' not found in module '{network_module_name}'."
        )

    return network_class


class BaseModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = params
        self.device = device
        # self.init_losses()

    @abstractmethod
    def compute_loss(self):
        pass
