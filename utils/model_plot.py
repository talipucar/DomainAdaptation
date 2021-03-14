"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Plot utilities. Used to plot losses recarded during training.
"""

import matplotlib.pyplot as plt


def save_loss_plot(losses, plots_path):
    x_axis = list(range(len(losses["rloss_e"])))
    plt.plot(x_axis, losses["rloss_e"], c='r', label="Training")
    title = "Training"
    if len(losses["vloss_e"]) >= 1:
        # If validation loss is recorded less often, we need to adjust x-axis values by the factor of difference
        beta = len(losses["rloss_e"]) / len(losses["vloss_e"])
        x_axis = list(range(len(losses["vloss_e"])))
        # Adjust the values of x-axis by beta factor
        x_axis = [beta*i for i in x_axis]
        plt.plot(x_axis, losses["vloss_e"], c='b', label="Validation")
        title += " and Validation "
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title(title + " Loss", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_path + "/loss.png")


