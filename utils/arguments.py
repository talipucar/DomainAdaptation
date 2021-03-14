"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: - Collects arguments from command line, and loads configuration from the yaml files.
             - Prints a summary of all options and arguments.
"""

import os
import pprint
import torch as th
from argparse import ArgumentParser
from os.path import dirname, abspath
from utils.utils import get_runtime_and_model_config, print_config

def get_arguments():
    # Initialize parser
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-d1", "--dataset1", type=str, default="transcriptomics_sputum_v1")
    parser.add_argument("-d2", "--dataset2", type=str, default="transcriptomics_blood_v1")
    parser.add_argument("-d3", "--dataset3", type=str, default="transcriptomics_bronchila_v1")
    # Input image size if images are being used
    parser.add_argument("-img", "--image_size", type=int, default=64)
    # Input channel size if images are being used
    parser.add_argument("-ch", "--channel_size", type=int, default=1)
    # Whether to use GPU.
    parser.add_argument("-g", "--gpu", dest='gpu', action='store_true')
    parser.add_argument("-ng", "--no_gpu", dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-cuda", "--cuda_number", type=str, default='0')
    # Experiment number if MLFlow is on
    parser.add_argument("-ex", "--experiment", type=int, default=1)
    # Return parser arguments
    return parser.parse_args()

def get_config(args):
    # Get path to the root
    root_path = dirname(abspath(__file__))
    # Get path to the runtime config file
    config = os.path.join(root_path, "config", "runtime.yaml")
    # Load runtime config from config folder: ./config/ and flatten the runtime config
    config = get_runtime_and_model_config()
    # Copy dataset names to config to use later
    config["dataset1"], config["dataset2"], config["dataset3"] = args.dataset1, args.dataset2, args.dataset3
    # Copy image size argument to config to use later
    config["img_size"] = args.image_size
    # Copy channel size argument to config to modify default architecture in model config
    config["conv_dims"][0][0] = args.channel_size
    # Define which device to use: GPU or CPU
    config["device"] = th.device('cuda:'+args.cuda_number if th.cuda.is_available() and args.gpu else 'cpu')
    # Return
    return config

def print_config_summary(config, args):
    # Summarize config on the screen as a sanity check
    print(100*"=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100*"=")
    print(f"Arguments being used:\n")
    print_config(args)
    print(100*"=")