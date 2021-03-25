"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Utility functions for evaluations (used in 1_eval.py)
"""

import os
from os.path import dirname, abspath

import torch as th
import torch.utils.data

from utils.utils import tsne

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

torch.manual_seed(1)


def linear_model_eval_wrapper(config, Xdata, Xrecon, Xtran_to_ds1, Xtran_to_ds2, Xtran_to_ds3, clabels, dlabels_raw,
                              mode, save_files=False):
    # Extract samples in each domain from the input data
    Xds1, Xds2, Xds3 = get_translated_outputs(Xdata, dlabels_raw)

    # Extract cohort labels in each domain from the input labels
    clabels_ds1, clabels_ds2, clabels_ds3 = get_translated_outputs(clabels, dlabels_raw)

    # Extract samples in each domain that are translated to ds1
    Xds1_to_ds1, Xds2_to_ds1, Xds3_to_ds1 = get_translated_outputs(Xtran_to_ds1, dlabels_raw)

    # Extract samples in each domain that are translated to ds2
    Xds1_to_ds2, Xds2_to_ds2, Xds3_to_ds2 = get_translated_outputs(Xtran_to_ds2, dlabels_raw)

    # Extract samples in each domain that are translated to ds3
    Xds1_to_ds3, Xds2_to_ds3, Xds3_to_ds3 = get_translated_outputs(Xtran_to_ds3, dlabels_raw)

    # The Linear model is trained on the raw data, and tested on reconstructed samples.
    print(20 * "*" + "Classification test in data space" + 20 * "*")
    linear_model_eval(Xdata, dlabels_raw, Xrecon, dlabels_raw,
                      description="Trained on input original domains, tested on reconstructions")

    # Trained on domain 3, and tested on translation from domain 1 i.e. 1->3
    print(20 * "*" + "Classification test in data space" + 20 * "*")
    linear_model_eval(Xds3, clabels_ds3, Xds1_to_ds3, clabels_ds1,
                      description="Trained on input original Domain-3, tested on translations from Domain-1")

    # Trained on domain 3, and tested on translation from domain 2 i.e. 2->3
    print(20 * "*" + "Classification test in data space" + 20 * "*")
    linear_model_eval(Xds3, clabels_ds3, Xds2_to_ds3, clabels_ds2,
                      description="Trained on input original Domain-3, tested on translations from Domain-2")

    # Trained on domain 1, and tested on translation from domain 2 i.e. 2->1
    print(20 * "*" + "Classification test in data space" + 20 * "*")
    linear_model_eval(Xds1, clabels_ds1, Xds2_to_ds1, clabels_ds2,
                      description="Trained on input original Domain-1, tested on translations from Domain-2")

    # Trained on domain 2, and tested on translation from domain 1 i.e. 1->2
    print(20 * "*" + "Classification test in data space" + 20 * "*")
    linear_model_eval(Xds2, clabels_ds2, Xds1_to_ds2, clabels_ds1,
                      description="Trained on input original Domain-2, tested on translations from Domain-1")

    # Save translation from domain 1 -> 2, and 2 -> 1 to use them later
    if save_files:
        print(f"Saving ds1->ds2 and ds2->ds1 translations as csv files...")
        cl_path = "./results/" + config["framework"] + "/evaluation/clusters/"
        save_np2csv([Xds1_to_ds2, clabels_ds1], save_as=cl_path + "/ds1_to_ds2_" + mode + ".csv")
        save_np2csv([Xds2_to_ds1, clabels_ds2], save_as=cl_path + "/ds2_to_ds1_" + mode + ".csv")
        save_np2csv([Xrecon, dlabels_raw], save_as=cl_path + "/xrecon_dlabels_" + mode + ".csv")
        save_np2csv([Xrecon, clabels], save_as=cl_path + "/xrecon_clabels_" + mode + ".csv")
        save_np2csv([Xdata, dlabels_raw], save_as=cl_path + "/Xinput_dlabels_" + mode + ".csv")
        save_np2csv([Xdata, clabels], save_as=cl_path + "/Xinput_clabels_" + mode + ".csv")


def translate_to_new_domain(autoencoder, config, Xdata, Xtran_to_ds3_l, ztran_to_ds3_l, to_domain=2):
    # Create labels to translate all domains to one particular domain
    dlabels_for_translation = domain_labels_for_tranlation(config, domain_label=[to_domain])

    # Prepare input data for translation to a particular domain
    input_data_trans = [Xdata, dlabels_for_translation] if config["conditional"] else Xdata

    # 1st Forward pass on the Autoencoder for translation to specific domain
    Xrecon1, _, _, _ = autoencoder(input_data_trans)
    # 2nd Forward pass on the Autoencoder to get translations in latent space
    _, zt, _, _ = autoencoder([Xrecon1, dlabels_for_translation])

    # Save translations in reconstruction and latent space to the lists
    Xrecon1 = Xrecon1.cpu().numpy()
    zt = zt.cpu().numpy()

    Xtran_to_ds3_l.append(Xrecon1)
    ztran_to_ds3_l.append(zt)

    return Xtran_to_ds3_l, ztran_to_ds3_l


def linear_model_eval(X_train, y_train, X_test, y_test, use_scaler=False, description="Baseline: PCA + Logistic Reg."):
    """
    :param ndarray X_train:
    :param list y_train:
    :param ndarray X_test:
    :param list y_test:
    :param bool use_scaler:
    :param str description:
    :return:
    """
    # Initialize Logistic regression
    clf = RandomForestClassifier(
        n_estimators=100)  # LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=0.1)
    # Fit model to the data
    clf.fit(X_train, y_train)
    # Summary of performance
    print(10 * ">" + description)
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))


def plot_clusters(config, z, ztran_to_ds3, clabels, dlabels_raw, plot_suffix="_inLatentSpace"):
    # Number of columns for legends, where each column corresponds to a cluster/cohort
    ncol = len(list(set(clabels)))
    # dlegends = ["1", "2", "3", "4", "5", "6", "7", ...]
    dlegends = [str(i + 1) for i in range(len(list(set(dlabels_raw))))]
    # clegends = ["A", "B", "C", "D", ...]..choose first ncol characters, one per coluster
    clegends = list("ABCDEFGH")[0:ncol]
    # Create new labels so that we can show cohorts and domains together. These labels will be used as keys to a dict to map to legend names
    new_labels = 4 * clabels + dlabels_raw

    # Show domains only
    visualise_clusters(config, z, dlabels_raw, plt_name="domains" + plot_suffix, legend_title="Domains",
                       legend_labels=dlegends)
    # Show cohorts only
    visualise_clusters(config, z, clabels, plt_name="cohorts" + plot_suffix, legend_title="Cohorts",
                       legend_labels=clegends)
    # Show both cohorts and domains
    visualise_clusters(config, z, new_labels, plt_name="domain_cohorts" + plot_suffix, alpha=1.0,
                       legend_title="Cohorts/Domains", ncol=ncol)

    # Plot translations from all domains to a particular domain (e.g. ds3)
    # Show both cohorts and domains
    visualise_clusters(config, ztran_to_ds3, new_labels, plt_name="domains_cohorts_translations" + plot_suffix,
                       alpha=1.0, legend_title="Cohorts/Domains", ncol=ncol)
    # Show domains only
    visualise_clusters(config, ztran_to_ds3, dlabels_raw, plt_name="domains_translations" + plot_suffix,
                       legend_title="Domains", legend_labels=dlegends)


def visualise_clusters(config, embeddings, labels, plt_name="test", alpha=1.0, legend_title=None, legend_labels=None,
                       ncol=1):
    """
    :param ndarray embeddings: Latent representations of samples.
    :param ndarray labels: Class labels;
    :param plt_name: Name to be used when saving the plot.
    :return: None
    """
    # Define colors to be used for each class/cluster/cohort
    color_list = ['#66BAFF', '#FFB56B', '#8BDD89', '#faa5f3', '#fa7f7f',
                  '#008cff', '#ff8000', '#04b000', '#de4bd2', '#fc3838',
                  '#004c8b', "#964b00", "#026b00", "#ad17a1", '#a80707',
                  "#00325c", "#e41a1c", "#008DF9", "#570950", '#732929']

    color_list2 = ['#66BAFF', '#008cff', '#004c8b', '#00325c',
                   '#FFB56B', '#ff8000', '#964b00', '#e41a1c',
                   '#8BDD89', "#04b000", "#026b00", "#008DF9",
                   "#faa5f3", "#de4bd2", "#ad17a1", "#570950",
                   '#fa7f7f', '#fc3838', '#a80707', '#732929']

    # If there are more than 3 types of labels, we want to plot both cohort, and domains, so change color scheme.
    color_list = color_list2 if len(list(set(labels))) > config["n_cohorts"] + 1 else color_list

    # Map class to legend texts. "A1" = Cohort-A in Domain-1
    c2l = {"0": "A1", "1": "A2", "2": "A3", "3": "A4",
           "4": "B1", "5": "B2", "6": "B3", "7": "B4",
           "8": "C1", "9": "C2", "10": "C3", "11": "C4",
           "12": "D1", "13": "D2", "14": "D3", "15": "D4",
           "16": "E1", "17": "E2", "18": "E3", "19": "E4", }

    # Used to adjust space for legends based on number of columns in the legend. ncol: subplot_adjust
    legend_space_adjustment = {"1": 0.9, "2": 0.9, "3": 0.75, "4": 0.65, "5": 0.65}

    # Initialize an empty dictionary to hold the mapping for color palette
    palette = {}
    # Map colors to the indexes.
    for i in range(len(color_list)):
        palette[str(i)] = color_list[i]
    # Make sure that the labels are 1D arrays
    y = labels.reshape(-1, )
    # Turn labels to a list
    y = list(map(str, y.tolist()))
    # Define number of sub-plots to draw. In this case, 2, one for PCA, and one for t-SNE
    img_n = 2
    # Initialize subplots
    fig, axs = plt.subplots(1, img_n, figsize=(12, 3.5), facecolor='w', edgecolor='k')
    # Adjust the whitespace around sub-plots
    fig.subplots_adjust(hspace=.1, wspace=.1)
    # adjust the ticks of axis.
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',
        left=False,  # both major and minor ticks are affected
        right=False,
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    # Flatten axes if we have more than 1 plot. Or, return a list of 2 axs to make it compatible with multi-plot case.
    axs = axs.ravel() if img_n > 1 else [axs, axs]

    # Get 2D embeddings, using PCA
    pca = PCA(n_components=2)
    # Fit training data and transform
    embeddings_pca = pca.fit_transform(embeddings)  # if embeddings.shape[1]>2 else embeddings
    # Set the title of the sub-plot
    axs[0].title.set_text('Embeddings from PCA')
    # Plot samples, using each class label to define the color of the class.
    sns_plt = sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], ax=axs[0], palette=palette, hue=y, s=20,
                              alpha=alpha)
    # Overwrite legend labels
    overwrite_legends(sns_plt, c2l, fig, ncol=ncol, title=legend_title, labels=legend_labels)
    # Get 2D embeddings, using t-SNE
    embeddings_tsne = tsne(embeddings)  # if embeddings.shape[1]>2 else embeddings
    # Set the title of the sub-plot
    axs[1].title.set_text('Embeddings from t-SNE')
    # Plot samples, using each class label to define the color of the class.
    sns_plt = sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], ax=axs[1], palette=palette, hue=y, s=20,
                              alpha=alpha)
    # Overwrite legend labels
    overwrite_legends(sns_plt, c2l, fig, ncol=ncol, title=legend_title, labels=legend_labels)
    # Remove legends in sub-plots
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=legend_space_adjustment[str(ncol)])

    # Get the path to the project root
    root_path = os.path.dirname(os.path.dirname(__file__))
    # Define the path to save the plot to.
    fig_path = os.path.join(root_path, "results", config["framework"], "evaluation", "clusters", plt_name + ".png")
    # Define tick params
    plt.tick_params(axis=u'both', which=u'both', length=0)
    # Save the plot
    plt.savefig(fig_path, bbox_inches="tight")
    # Clear figure just in case if there is a follow-up plot.
    plt.clf()


def overwrite_legends(sns_plt, c2l, fig, ncol, title=None, labels=None):
    # Get legend handles and labels
    handles, legend_txts = sns_plt.get_legend_handles_labels()
    # Turn str to int before sorting ( to avoid wrong sort order such as having '10' in front of '4' )
    legend_txts = [int(d) for d in legend_txts]
    # Sort both handle and texts so that they show up in a alphabetical order on the plot
    legend_txts, handles = (list(t) for t in zip(*sorted(zip(legend_txts, handles))))
    # Turn int to str before using labels
    legend_txts = [str(i) for i in legend_txts]
    # Get new legend labels using class-to-label map
    new_labels = [c2l[legend_text] for legend_text in legend_txts]
    # Overwrite new_labels if it is given by user.
    new_labels = labels or new_labels
    # Define the figure title
    title = title or "Cohorts/Domains"
    # Overwrite the legend labels and add a title to the legend
    fig.legend(handles, new_labels, loc="center right", borderaxespad=0.1, title=title, ncol=ncol)
    sns_plt.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    sns_plt.tick_params(top=False, bottom=False, left=False, right=False)


def domain_labels_for_tranlation(options, domain_label=[2]):
    # Assign each domain to a label: domain-1:0, domain-2:1 and so on.
    domain_labels = []
    # Repeat each for number of batch size so that we have label for each data point from each domain
    domain_labels = options["n_domains"] * options["batch_size"] * domain_label
    # Turn labels to torch tensor
    domain_labels = th.from_numpy(np.array(domain_labels))
    # Turn them into one-hot embeddings, shape: (3 x batch_size, number of domains)
    y = th.eye(options["n_domains"])
    # Return one-hot encoded domain labels
    return y[domain_labels].to(options["device"])


def save_np2csv(np_list, save_as="test.csv"):
    # Get numpy arrays and label lists
    Xtr, ytr = np_list
    # Turn label lists into numpy arrays
    ytr = np.array(ytr, dtype=np.int8)
    # Get column names
    columns = ["label"] + list(map(str, list(range(Xtr.shape[1]))))

    # Concatenate "scaled" features and labels
    data_tr = np.concatenate((ytr.reshape(-1, 1), Xtr), axis=1)
    # Generate new dataframes with "scaled features" and labels
    df_tr = pd.DataFrame(data=data_tr, columns=columns)
    # Show samples from scaled data
    print("Samples from the dataframe:")
    print(df_tr.head())
    # Save the dataframe as csv file
    df_tr.to_csv(save_as, index=False)
    # Print an informative message
    print(f"The dataframe is saved as {save_as}")


def append_tensors_to_lists(list_of_lists, list_of_tensors):
    # Go through each tensor and corresponding list
    for i in range(len(list_of_tensors)):
        # Convert tensor to numpy and append it to the corresponding list
        list_of_lists[i] += [list_of_tensors[i].cpu().numpy()]
    # Return the lists
    return list_of_lists


def concatenate_lists(list_of_lists):
    list_of_np_arrs = []
    # Pick a list of numpy arrays ([np_arr1, np_arr2, ...]), concatenate numpy arrs to a single one (np_arr_big),
    # and append it back to the list ([np_arr_big1, np_arr_big2, ...])
    for list_ in list_of_lists:
        list_of_np_arrs.append(np.concatenate(list_))
    # Return numpy arrays
    return list_of_np_arrs


def get_translated_outputs(Xtran, dlabels_raw):
    return Xtran[dlabels_raw == 0], Xtran[dlabels_raw == 1], Xtran[dlabels_raw == 2]

