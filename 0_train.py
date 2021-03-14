"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Trains an Autoencoder (AE) in adversarial setting for domain adaptation and translation 
using different multiple sources of data with same number of features.

An AE is initialized together with two discriminators, one for latent samples, and one for reconstructed samples. 
It is trained with adversarial loss, during which it aligns the latent spaces of different domains. 

"""

import time
import mlflow
from src.model import AEModel
from utils.load_data import Loader
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims

def train(config, data_loader, save_weights=True):
    """
    :param dict config: Dictionary containing options.
    :param IterableDataset data_loader: Pytorch data loader.
    :param bool save_weights: Saves model if True.
    :return:

    Utility function for training and saving the model.
    """
    # Instantiate model
    model = AEModel(config)
    # Start the clock to measure the training time
    start = time.process_time()
    # Fit the model to the data
    model.fit(data_loader)
    # Total time spent on training
    training_time = time.process_time() - start
    # Report the training time
    print(f"Training time:  {training_time//60} minutes, {training_time%60} seconds")
    # Save the model for future use
    _ = model.save_weights() if save_weights else None
    print("Done with training...")
    # Track results
    if config["mlflow"]:
        # Log config with mlflow
        mlflow.log_artifacts("./config", "config")
        # Log model and results with mlflow
        mlflow.log_artifacts("./results/training/" + config["model_mode"], "training_results")
        # log model
        mlflow.pytorch.log_model(model, "models")

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
    # Start training and save model weights at the end
    train(config, [ds1_loader, ds2_loader, ds3_loader], save_weights=True)

if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
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

