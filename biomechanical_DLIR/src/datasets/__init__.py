"""
BaseDataset and Dataset Loader

Defines the abstract BaseDataset class for custom datasets and a dynamic loader function `load_dataset` 
that imports a dataset class based on its name from corresponding `<name>_dataset.py` files.

The BaseDataset handles shared setup logic (params, transforms, and data dict), 
while requiring subclasses to implement core dataset methods.
"""

import os
import importlib.util
from abc import abstractmethod

# Directory containing dataset files
datasets_directory = os.path.dirname(__file__)

def load_dataset(dataset_name):
    """
    Load the dataset class of the specified dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        dataset_class: Class of the dataset.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        AttributeError: If the dataset class is not found.
    """

    dataset_file_path = os.path.join(datasets_directory, f"{dataset_name.lower()}_dataset.py")
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(f"Dataset file '{dataset_name.lower()}_dataset.py' not found in '{datasets_directory}'.")

    dataset_module_name = f"datasets.{dataset_name.lower()}_dataset"
    dataset_module_spec = importlib.util.spec_from_file_location(
                                    dataset_module_name,
                                    dataset_file_path
    )
    dataset_module = importlib.util.module_from_spec(dataset_module_spec)
    dataset_module_spec.loader.exec_module(dataset_module)

    dataset_class_name = dataset_name + 'Dataset'
    dataset_class = getattr(dataset_module, dataset_class_name, None)

    if dataset_class is None:
        raise AttributeError(f"Dataset class '{dataset_class_name}' not found in module '{dataset_module_name}'.")

    return dataset_class


from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,params):
        super().__init__()
        self.params = params
        self.transforms = self.init_transforms()
        self.data_dict = self.make_data_dict()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def init_transforms(self):
        pass

    @abstractmethod
    def make_data_dict(self):
        pass

