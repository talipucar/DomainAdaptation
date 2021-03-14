"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Evaluation of Domain Alignment & Translation using Autoencoders.
"""
import os
from os.path import dirname, abspath
import imageio
from tqdm import tqdm

import torch as th
import torch.utils.data

from src.model import AEModel
from utils.load_data import Loader
from sklearn.preprocessing import StandardScaler
from utils.arguments import print_config_summary
from utils.arguments import get_arguments, get_config
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims, tsne
from utils.eval_utils import linear_model_eval_wrapper, plot_clusters, translate_to_new_domain, append_tensors_to_lists, concatenate_lists

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import mlflow
torch.manual_seed(1)


def eval(data_loader, config):
    """
    :param IterableDataset data_loader: Pytorch data loader.
    :param dict config: Dictionary containing options.
    :return: None
    """
    # Instantiate Autoencoder model
    model = AEModel(config)
    # Load the model
    model.load_models()
    # Evaluate Autoencoder
    with th.no_grad():
        evalulate_models(data_loader, model, config, plot_suffix="test", mode='test')
        evalulate_models(data_loader, model, config, plot_suffix="train", mode='train')


def evalulate_models(data_loader, model, config, plot_suffix = "_Test", mode='train'):
    """
    :param IterableDataset data_loader: Pytorch data loader.
    :param model: Pre-trained autoencoder class.
    :param dict config: Dictionary containing options.
    :param plot_suffix: Custom suffix to use when saving plots.
    :return: None.
    """
    # Print whether we are evaluating training set, or test set
    print(f"{100*'#'}\n{100*'#'}")
    print(f"Evaluating on " + plot_suffix + " set...")
    # Print domain names
    print(f"{100*'='}\n{100*'='}")
    print(f"Domains evaluated: \n Domain-1: {config['dataset1']}, \n Domain-2: {config['dataset2']}, \n Domain-3: {config['dataset3']}")
    print(f"{100*'='}\n{100*'='}")
    
    # Get Autoencoders for both modalities
    autoencoder = model.autoencoder
    # Move the models to the device
    autoencoder.to(config["device"])
    # Set models to evaluation mode
    autoencoder.eval()
    
    # Get data loaders.
    ds1_loader, ds2_loader, ds3_loader = data_loader
    # zip() both data loaders, and cycle the one with smaller dataset to go through all samples of longer one.
    zipped_data_loaders_tr = zip(ds1_loader.train_loader, ds2_loader.train_loader, ds3_loader.train_loader)
    # zip() both data loaders, and cycle the one with smaller dataset to go through all samples of longer one.
    zipped_data_loaders_te = zip(ds1_loader.test_loader, ds2_loader.test_loader, ds3_loader.test_loader)
    # Choose either training, or test data loader
    zipped_data_loaders = zipped_data_loaders_tr if mode=='train' else zipped_data_loaders_te
    # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
    train_tqdm = tqdm(enumerate(zipped_data_loaders), total=len(ds1_loader.train_loader), leave=True)
    
    # Create empty lists to hold data.
    Xd_l, Xr_l, z_l, clabels_l, dlabels_l = [], [], [], [], []
    
    # Create empty lists to hold translations.
    Xtran_to_ds1_l, ztran_to_ds1_l = [], [] 
    Xtran_to_ds2_l, ztran_to_ds2_l = [], [] 
    Xtran_to_ds3_l, ztran_to_ds3_l = [], [] 

    # Go through batches
    for i, (ds1_dict, ds2_dict, ds3_dict) in train_tqdm:
        Xdata, clabels, dlabels = model.process_batch(ds1_dict, ds2_dict, ds3_dict)
        # Prepare input data
        input_data = [Xdata, dlabels] if config["conditional"] else Xdata

        # Forward pass on the Autoencoder
        Xrecon, zo, _, _ = autoencoder(input_data)
        
        # Translate all input data to Domain-1
        Xtran_to_ds1_l, ztran_to_ds1_l = translate_to_new_domain(autoencoder, config, Xdata, Xtran_to_ds1_l, ztran_to_ds1_l, to_domain=0)
        # Translate all input data to Domain-2
        Xtran_to_ds2_l, ztran_to_ds2_l = translate_to_new_domain(autoencoder, config, Xdata, Xtran_to_ds2_l, ztran_to_ds2_l, to_domain=1)
        # Translate all input data to Domain-3
        Xtran_to_ds3_l, ztran_to_ds3_l = translate_to_new_domain(autoencoder, config, Xdata, Xtran_to_ds3_l, ztran_to_ds3_l, to_domain=2)
        
        # Turn one-hot labels to raw labels ( e.g. 1, 2, 3 , 4, ...)
        dlabels = th.argmax(dlabels, dim=1)
        
        # Append tensors to the corresponding lists as numpy arrays
        Xd_l, Xr_l, z_l, clabels_l, dlabels_l = append_tensors_to_lists([Xd_l, Xr_l, z_l, clabels_l, dlabels_l], 
                                                                        [Xdata, Xrecon, zo, clabels, dlabels])
        
    # Turn list of numpy arrays to a single numpy array for input data, reconstruction, and latent samples.
    Xdata, Xrecon, z = concatenate_lists([Xd_l, Xr_l, z_l])
    # Turn list of numpy arrays to a single numpy array for cohort and domain labels (labels such as 1, 2, 3, 4,...).
    clabels, dlabels = concatenate_lists([clabels_l, dlabels_l])
    # Turn list of numpy arrays to a single numpy array for translations to reconstruction space
    Xtran_to_ds1, Xtran_to_ds2, Xtran_to_ds3 = concatenate_lists([Xtran_to_ds1_l, Xtran_to_ds2_l, Xtran_to_ds3_l])
    # Turn list of numpy arrays to a single numpy array for translations to latent space
    ztran_to_ds1, ztran_to_ds2, ztran_to_ds3 = concatenate_lists([ztran_to_ds1_l, ztran_to_ds2_l, ztran_to_ds3_l])

    # Visualise clusters
    plot_clusters(z, ztran_to_ds3, clabels, dlabels, plot_suffix="_inLatentSpace_" + plot_suffix)
    
    # The Linear model evaluation
    linear_model_eval_wrapper(Xdata, Xrecon, Xtran_to_ds1, Xtran_to_ds2, Xtran_to_ds3, clabels, dlabels, save_files=config["save_files"])
    print(f"Results are saved under ./results/evaluation/clusters/")
    
    
def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds1_loader = Loader(config, dataset_name=config["dataset1"])
    # Get data loader for second dataset.
    ds2_loader = Loader(config, dataset_name=config["dataset2"])
    # Get data loader for third dataset.
    ds3_loader = Loader(config, dataset_name=config["dataset3"])
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds1_loader, config)
    # Start evaluation
    eval([ds1_loader, ds2_loader, ds3_loader], config)
    

if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite batch size since test set might be very small (e.g. < 32)
    config["batch_size"] = 2
    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)
    # --If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        mlflow.set_experiment(experiment_name=config["model_mode"]+"_"+str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main with or without profiler
            run_with_profiler(main, config) if config["profile"] else main(config)
    else:
        # Run the main with or without profiler
        run_with_profiler(main, config) if config["profile"] else main(config)