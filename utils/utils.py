"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Utility functions
"""

import os
import imageio
import sys
import yaml
import numpy as np
from numpy.random import seed
import random as python_random
import cProfile, pstats
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from texttable import Texttable



def set_seed(options):
    seed(options["seed"])
    np.random.seed(options["seed"])
    python_random.seed(options["seed"])

    
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

        
def set_dirs(config):
    """
    It sets up directory that will be used to load processed_data and src as well as saving results.
    Directory structure:
          results -> processed_data: contains processed k-fold processed_data file.
                  -> src: contains saved src, trained on this database
                  -> results-> training_plots
    :return: None
    """
    # Set main results directory using database name. Exp:  processed_data/dpp19
    paths = config["paths"]
    # data > processed_data (This is disabled)
    # processed_data_dir = os.path.join(paths["data"], "processed_data")
    # results > training
    training_dir = os.path.join(paths["results"], "training")
    # results > evaluation
    evaluation_dir = os.path.join(paths["results"], "evaluation")
    # results > evaluation > clusters
    clusters_dir = os.path.join(evaluation_dir, "clusters")
    # results > evaluation > reconstruction
    recons_dir = os.path.join(evaluation_dir, "reconstructions")
    # results > training > model_mode = vae
    model_mode_dir = os.path.join(training_dir, config["model_mode"])
    # results > training > model_mode > model
    training_model_dir = os.path.join(model_mode_dir, "model")
    # results > training > model_mode > plots
    training_plot_dir = os.path.join(model_mode_dir, "plots")
    # results > training > model_mode > loss
    training_loss_dir = os.path.join(model_mode_dir, "loss")
    # Create any missing directories
    # if not os.path.exists(processed_data_dir):
    #     os.makedirs(processed_data_dir)
    if not os.path.exists(training_model_dir):
        os.makedirs(training_model_dir)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    if not os.path.exists(clusters_dir):
        os.makedirs(clusters_dir)
    if not os.path.exists(recons_dir):
        os.makedirs(recons_dir)
    if not os.path.exists(model_mode_dir):
        os.makedirs(model_mode_dir)
    if not os.path.exists(training_model_dir):
        os.makedirs(training_model_dir)
    if not os.path.exists(training_plot_dir):
        os.makedirs(training_plot_dir)
    if not os.path.exists(training_loss_dir):
        os.makedirs(training_loss_dir)
    # Print a message.
    print("Directories are set.")

def get_runtime_and_model_config():
    try:
        with open("./config/runtime.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading runtime config file")
    # Update the config by adding the model specific config to runtime config
    config = update_config_with_model(config)
    return config


def update_config_with_model(config):
    model_config = config["model_config"]
    try:
        with open("./config/"+model_config+".yaml", "r") as file:
            model_config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading model config file")
    config.update(model_config)
    return config


def update_config_with_model_dims(data_loader, config):
    # Get the first batch (data is in dictionary format)
    d = next(iter(data_loader.train_loader))
    # Get the features and turn them into numpy.
    xi = d['tensor'].cpu().numpy()
    # Get the number of features
    dim = xi.shape[-1]
    # Update the dims of model architecture by adding the number of features as the first dimension
    config["dims"].insert(0, dim)
    return config


def run_with_profiler(main_fn, config):
    profiler = cProfile.Profile()
    profiler.enable()
    # Run the main
    main_fn(config)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()
    
    
def tsne(latent):
    """
    :param latent: Embeddings to use.
    :return: 2D embeddings
    """
    mds = manifold.TSNE(n_components=2, init='pca', random_state=0)
    return mds.fit_transform(latent)


def generate_image(inputs, name="image"):
    """
    :param inputs: Images to be plotted.
    :param name:  Name to be given to the plot.
    :return:
    """
    img_dir = os.path.join("./results/evaluation/", "reconstructions")
    os.makedirs(img_dir, exist_ok=True)
    imageio.imwrite(os.path.join(img_dir, name + ".jpg"), np.uint8(inputs.reshape(64, 64) * 255))


def scale_data(Xtrain, Xtest):
    """
    :param Xtrain:
    :param Xtest:
    :return:
    """
    # Initialize scaler
    scaler = StandardScaler()
    # Fit and transform representations from training set
    Xtrain = scaler.fit_transform(Xtrain)
    # Transform representations from test set
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest


def print_config(args):
    """
    Prints the YAML config and ArgumentParser arguments in a neat format.
    :param args: Parameters/config used for the model.
    """
    # Yaml config is a dictionary while parser arguments is an object. Use vars() only on parser arguments.
    if type(args) is not dict:
        args = vars(args)
    # Sort keys
    keys = sorted(args.keys())
    # Initialize table
    table = Texttable() 
    # Add rows to the table under two columns ("Parameter", "Value").
    table.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    # Print the table.
    print(table.draw())