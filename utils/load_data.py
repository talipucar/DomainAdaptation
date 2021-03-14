"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: A library for data loaders.
"""

import os
import cv2
from skimage import io

import numpy as np
import pandas as pd
import datatable as dt

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



class Loader(object):
    """
    Author: Talip Ucar
    Email: ucabtuc@gmail.com
    """
    def __init__(self, config, dataset_name, eval_mode=False, kwargs={}):
        """
        :param dict config: Configuration dictionary.
        :param str dataset_name: Name of the dataset to use.
        :param bool eval_mode: Whether the dataset is used for evaluation. False by default.
        :param dict kwargs: Additional parameters if needed.
        """
        # Get batch size
        batch_size = config["batch_size"]
        # Get config
        self.config = config
        # Set main results directory using database name. Exp:  processed_data/dpp19
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name)
        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(dataset_name, file_path, eval_mode=eval_mode)
        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    def get_dataset(self, dataset_name, file_path, eval_mode=False):
        # Create dictionary for loading functions of datasets. 
        # If you add a new dataset, add its corresponding dataset class here in the form 'dataset_name': ClassName
        loader_map = {'default_loader': TabularDataset}
        # Get dataset. Check if the dataset has a custom class. If not, then assume a tabular data with labels in the first column
        dataset = loader_map[dataset_name] if dataset_name in loader_map.keys() else loader_map['default_loader']
        # Transformation for training dataset. If we are evaluating the model, use ToTensorNormalize.
        train_transform = ToTensorNormalize()
        # Training and Validation datasets
        train_dataset = dataset(datadir=file_path, dataset_name=dataset_name, mode='train', transform=train_transform)
        # Test dataset
        test_dataset = dataset(datadir=file_path, dataset_name=dataset_name, mode='test')
        # Return
        return train_dataset, test_dataset


class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # Assumes that min-max scaling is done when pre-processing the data (i.e. not here)
        return torch.from_numpy(sample).float()


class TabularDataset(Dataset):
    def __init__(self, datadir, dataset_name, mode='train', transform=ToTensorNormalize()):
        """
        Expects two csv files with _tr and _te suffixes for training and test datasets.
        Example: dataset_name_tr.csv, dataset_name_te.csv
        """
        self.datadir = datadir
        self.dataset_name = dataset_name
        self.mode = mode
        self.data, self.labels = self._load_data()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        cluster = self.labels[idx]
        
        if self.transform:
            # transform the tensor 
            sample = self.transform(sample)
        
        return {'tensor': sample, 'binary_label': int(cluster)}

    def _load_data(self):
        # Get the file name, ignoring suffix e.g. 'test_dataset' from 'test_dataset_tr.csv'
        file = self.dataset_name
        # Add suffix depending on whehter you want training set, or test
        file += "_tr.csv" if self.mode == 'train' else "_te.csv"
        # Load dataset
        data = dt.fread(os.path.join(self.datadir, file)).to_pandas()
        # Get data as numpy array
        data = data.values
        # Extract labels - assumes they are in the first column
        labels = data[:, 0]
        # Check the lowest label number (whether it is 0 or 1). And make sure that it is 0.
        labels = np.abs(labels - 1) if np.min(labels) == 1 else np.abs(labels)
        # Return features, and labels
        return data[:, 1:], labels