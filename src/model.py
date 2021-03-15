"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Class to train an Autoencoder using multiple datasets with same number of features and 
to align their latent representations. It is configured such that it expects 3 datasets at the moment. However, 
extending it to more or less data sources should be trivial.

Two discriminators are used:
    I) A discriminator is used to align corresponding clusters across different data sources in the latent space.
    Clusters are aligned by using a mixture of Gaussians

    II) A discriminator is used to compare reconstructions at the output of Autoencoder and original samples. This
    is to improve the quality of reconstructions.
    
TODO: Making number of datasets being used a flexible choice rather than fixing it to three.

"""

import os
import gc

import itertools
from itertools import cycle
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils.utils import set_seed, set_dirs
from utils.loss_functions import get_vae_loss, getMSEloss, get_generator_loss, get_discriminator_loss
from utils.model_plot import save_loss_plot
from utils.model_utils import Autoencoder, Discriminator, Classifier

import torch as th
import torch.nn.functional as F

th.autograd.set_detect_anomaly(True)


class AEModel:
    """
    Model: Consists of an Autoencoder together with two Discriminators, one for latent, and one for reconstructions.
    Loss function: Reconstruction loss of untrained Autoencoder + either Adversarial losses for two Discriminators.
    ------------------------------------------------------------
    Architecture:  X -> Encoder -> z -> Decoder -> X' __ Discriminator
                                   |               X  _/
                                   |
                                    \_ Discriminator
                     GMM -> z_prior /
                   
    ------------------------------------------------------------
    Autoencoders can be configured as
                        - Autoencoder (ae),
                        - Variational autoencoder (vae),
                        - Beta-VAE (bvae),
                        - Adversarial autoencoder (aae).
    ------------------------------------------------------------
    Autoencoder can have a CNN-based architecture, or fully-connected one. Use "convolution=true" in model config file if
    CNN-based model should be used.
    
    Dictionary:
    clabels = cohort labels, or cluster labels within each domain (data source)
    dlabels = domain labels (labels assigned to each data source)
    ds = data source, or data set. For example, ds1 = data source 1
    ae = AutoEncoder
    aae = Advarserial AutoEncoder
    disc = discriminator
    gen = generator
    recon = reconstruction
    """

    def __init__(self, options):
        """Class to train an autoencoder model with two discriminator for domain adaptation/tranlation/alignment.

        Args:
            options (dict): Configuration dictionary.
        """
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # Set paths for results and Initialize some arrays to collect data during training
        self._set_paths()
        # Set directories i.e. create ones that are missing.
        set_dirs(self.options)
        # ------Network---------
        # Instantiate networks
        print("Building the models for Data Alignment and Translation...")
        # Set Autoencoder i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_autoencoder()
        # If supervised, use discriminators
        if self.options["adv_training"]:
            # Set AEE i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
            self.set_aae()
            # Assign domains to labels e.g. label: 0, 1, 2 for 3 data sources.
            self.set_domain_labels()
        # Set scheduler (its use is optional)
        self._set_scheduler()
        # Print out model architecture
        self.print_model_summary()

    def set_autoencoder(self):
        """Sets up the autoencoder model, optimizer, and loss"""
        # Instantiate the model for the Autoencoder 
        self.autoencoder = Autoencoder(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"autoencoder": self.autoencoder})
        # Assign autoencoder to a device
        self.autoencoder.to(self.device)
        # Reconstruction loss
        self.recon_loss = getMSEloss
        # Set optimizer for autoencoder
        self.optimizer_ae = self._adam([self.autoencoder.parameters()], lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"recon_loss": [], "kl_loss": []})

    def set_aae(self):
        """Sets up the discriminator models, optimizer, and loss"""
        # Instantiate Discriminators for latent space
        self.discriminator_z = Discriminator(self.options, input_dim=self.options["dims"][-1]+self.options["n_cohorts"])
        # Instantiate Discriminators for data space
        self.discriminator_x = Discriminator(self.options, input_dim=self.options["dims"][0]+self.options["n_domains"])
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"discriminator_z": self.discriminator_z, "discriminator_x": self.discriminator_x})
        # Assign models to the device
        _ = [model.to(self.device) for model in [self.discriminator_z, self.discriminator_x]]
        # Generator loss
        self.gen_loss = get_generator_loss
        # Discriminator loss for latent and data space
        self.disc_loss = get_discriminator_loss
        # Set optimizer for generator for latent space
        self.optimizer_gen_z = self._adam([self.autoencoder.encoder.parameters()], lr=1e-3)
        # Set optimizer for generator for data space
        self.optimizer_gen_x = self._adam([self.autoencoder.decoder.parameters()], lr=1e-3)
        # Set optimizer for discriminator of latent space
        self.optimizer_disc_z = self._adam([self.discriminator_z.parameters()], lr=1e-5)
        # Set optimizer for discriminator of data space
        self.optimizer_disc_x = self._adam([self.discriminator_x.parameters()], lr=1e-5)
        # Add items to summary to be used for reporting later
        self.summary.update({"disc_z_train_acc": [], "disc_z_test_acc": []})

    def set_parallelism(self, model):
        """NOT USED - Sets up parallelism in training."""
        # If we are using GPU, and if there are multiple GPUs, parallelize training
        if th.cuda.is_available() and th.cuda.device_count() > 1:
            print(th.cuda.device_count(), " GPUs will be used!")
            model = th.nn.DataParallel(model)
        return model

    def fit(self, data_loaders):
        """Fits model to the data

        Args:
            data_loaders (list): List of dataloaders for multiple datasets.
        """
        # Get data loaders for three datasets
        ds1_loader, ds2_loader, ds3_loader = data_loaders

        # Placeholders for record batch losses
        self.loss = {"rloss_b": [], "rloss_e": [], "kl_loss": [], "vloss_e": [], "aae_loss_z": [], "aae_loss_x": []}

        # Turn on training mode for each model.
        self.set_mode(mode="training")
        
        # Compute total number of batches per epoch
        self.total_batches = len(ds1_loader.train_loader)
        
        # Start joint training of Autoencoder, and/or classifier
        for epoch in range(self.options["epochs"]):
            
            # Change learning rate if schedular=True
            _ = self.scheduler.step() if self.options["scheduler"] else None
            
            # zip() both data loaders, and cycle the one with smaller dataset to go through all samples of longer one.
            zipped_data_loaders = zip(ds1_loader.train_loader, ds2_loader.train_loader, cycle(ds3_loader.train_loader))
            
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(enumerate(zipped_data_loaders), total=self.total_batches, leave=True)
            
            # Go through batches
            for i, (ds1_dict, ds2_dict, ds3_dict) in self.train_tqdm:
                
                # Get features, labels in each dataset, and labels assigned to domains
                Xdata, labels, dlabels = self.process_batch(ds1_dict, ds2_dict, ds3_dict)
                
                # 0 - Update Autoencoder
                self.update_autoencoder(Xdata, dlabels)
                
                if self.options["adv_training"]:
                    # Update generator (Encoder) and discriminators in z- and x- space
                    # 1 - In z-space, update generator and discriminator
                    self.update_generator_discriminator_z([Xdata, labels, dlabels])

                    # 2 - Forward pass on the Autoencoder
                    _, z, _, _ = self.autoencoder([Xdata, dlabels])
                    # In x-space, update generator and discriminator
                    self.update_generator_discriminator_x([Xdata, z, dlabels])

                    # 3 - Shuffle the data and label to update the parameters again to learn translations between domains
                    Xdata_shuffled, dlabels_shuffled = self.shuffle_tensors([Xdata, dlabels])

                    # 4 - Update generator and discriminator
                    self.update_generator_discriminator_x([Xdata_shuffled, z, dlabels_shuffled])
                
                # 5 - Update log message using epoch and batch numbers
                self.update_log(epoch, i)
                
                # 6 - Clean-up for efficient memory use.
                gc.collect()
            
            # Validate every nth epoch. n=1 by default
            if epoch % self.options["nth_epoch"] == 0:
                # Compute total number of batches, assuming all test sets have same number of samples
                total_val_batches = len(ds1_loader.test_loader)
                # Zip all test data loaders
                zipped_val_loaders = zip(ds1_loader.test_loader,ds2_loader.test_loader,ds3_loader.test_loader)
                # Compute validation loss
                _ = self.validate(zipped_val_loaders, total_val_batches) 
        
            # Get reconstruction loss for training per epoch
            self.loss["rloss_e"].append(sum(self.loss["rloss_b"][-self.total_batches:-1]) / self.total_batches)
            
        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")

    def validate(self, validation_loader, total_batches):
        """Computes validation loss.

        Args:
            validation_loader (): data loader for validation set.
            total_batches (int): total number of batches in validation set.

        Returns:
            float: validation loss
        """
        with th.no_grad():
            # Initialize validation loss
            vloss = 0
            # Turn on evaluatin mode
            self.set_mode(mode="evaluation")
            # Print  validation message
            print(f"Computing validation loss. #Batches:{total_batches}")
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            val_tqdm = tqdm(enumerate(validation_loader), total=total_batches, leave=True)
            # Go through batches
            for i, (ds1_dict, ds2_dict, ds3_dict) in val_tqdm:
                # Get features, labels in each dataset, and labels assigned to domains
                Xdata, labels, dlabels = self.process_batch(ds1_dict, ds2_dict, ds3_dict)
                # Prepare input data
                input_data = [Xdata, dlabels] if self.options["conditional"] else Xdata
                recon, latent, _, _ = self.autoencoder(input_data)
                # Record validation loss
                val_loss = getMSEloss(recon, Xdata)
                # Get validation loss
                vloss = vloss + val_loss.item()
                # Clean up to avoid memory issues
                del val_loss, recon, latent
                gc.collect()
            # Turn on training mode
            self.set_mode(mode="training")
            # Compute mean validation loss
            vloss = vloss / total_batches
            # Record the loss
            self.loss["vloss_e"].append(vloss)
            # Return mean validation loss
        return vloss

    def update_autoencoder(self, Xdata, dlabels):
        """Updates autoencoder model.

        Args:
            Xdata (ndarray): 2D array containing data with float type
            dlabels (ndarray): 1D array containing data with int type
        """
        # Prepare input data
        input_data = [Xdata, dlabels] if self.options["conditional"] else Xdata
        # Forward pass on Autoencoder
        Xrecon, z, z_mean, z_logvar = self.autoencoder(input_data)
        # Compute reconstruction loss
        recon_loss = self.recon_loss(Xrecon, Xdata)
        # Add KL loss to compute total loss if we are using variational methods
        total_loss, kl_loss = get_vae_loss(recon_loss, z_mean, z_logvar, self.options)
        # Record reconstruction loss
        self.loss["rloss_b"].append(recon_loss.item())
        # Record KL loss if we are using variational inference
        self.loss["kl_loss"] += [kl_loss.item()] if self.options["model_mode"] in ["vae", "bvae"] else []
        # Update Autoencoder params
        self._update_model(total_loss, self.optimizer_ae, retain_graph=True)
        # Delete loss and associated graph for efficient memory usage
        del recon_loss, total_loss, kl_loss, Xrecon, z_mean, z_logvar

    def update_generator_discriminator_z(self, data, retain_graph=True):
        """Updates encoder and discriminator used for latent space.

        Args:
            data (list): List of ndarrays
            retain_graph (bool):

        Returns:
            None
        """
        # Get the output dimension of classifier
        num_classes = self.options["n_cohorts"]
        # Get the data: Xbatch: features, clabels=cohort labels, dlabels=domain labels
        Xbatch, clabels, dlabels = data
        # Sample real samples based on class proportions
        latent_real, labels_real = self.supervised_gaussian_mixture(clabels)
        latent_real, labels_real = shuffle(latent_real, labels_real)
        latent_real = th.from_numpy(latent_real).float().to(self.device)
        labels_real = th.from_numpy(labels_real).long().to(self.device)
        # Normalize the noise if samples from posterior (i.e. latent variable) is also normalized.
        latent_real = F.normalize(latent_real, p=2, dim=1) if self.options["normalize"] else latent_real
        
        # 1)----  Start of Discriminator update: Autoencoder in evaluation mode  ------------------------
        self.autoencoder.eval()
        # Forward pass on Autoencoder
        _, latent_fake, _, _ = self.autoencoder([Xbatch, dlabels])

        # Turn labels to one-hot encoded form
        clabels = self.one_hot_embedding(clabels, num_classes)
        labels_real = self.one_hot_embedding(labels_real, num_classes)

        # Concatenate cluster labels of data to its corresponding real latent samples (to make it conditional)
        latent_real = th.cat((latent_real, labels_real.float().view(-1, num_classes)), dim=1)
        # Concatenate cluster labels of data to its corresponding fake latent samples (to make it conditional)
        latent_fake = th.cat((latent_fake, clabels.float().view(-1, num_classes)), dim=1)

        # Get predictions for real samples
        pred_fake = self.discriminator_z(latent_fake.detach())
        # Get predictions for fake samples
        pred_real = self.discriminator_z(latent_real)
        # Compute discriminator loss
        disc_loss = self.disc_loss(pred_real, pred_fake)
        # Reset optimizer
        self.optimizer_disc_z.zero_grad()
        # Backward pass
        disc_loss.backward(retain_graph=retain_graph)
        # Update parameters of discriminator
        self.optimizer_disc_z.step()
        
        # 2)---- Start of Generator update: Autoencoder in train mode  ------------------------
        self.autoencoder.encoder.train()
        # Discriminator in evaluation mode
        self.discriminator_z.eval()
        #  Forward pass on Autoencoder
        _, latent_fake, _, _ = self.autoencoder([Xbatch, dlabels])
        # Concatenate cluster labels of data to its corresponding fake latent samples (to make it conditional)
        latent_fake = th.cat((latent_fake, clabels.float().view(-1, num_classes)), dim=1)
        # Get predictions for real samples
        pred_fake = self.discriminator_z(latent_fake)
        # Compute discriminator loss
        gen_loss = self.gen_loss(pred_fake)
        # Reset optimizer
        self.optimizer_gen_z.zero_grad()
        # Backward pass
        gen_loss.backward(retain_graph=retain_graph)
        # Update parameters of discriminator
        self.optimizer_gen_z.step()
        # Turn training mode back on. Default mode is training
        self.set_mode()
        # Record losses
        self.loss["aae_loss_z"].append([disc_loss.item(), gen_loss.item()])
        # Delete losses (graphs) for efficient memory usage
        self.clean_up_memory([disc_loss, gen_loss])

    def update_generator_discriminator_x(self, data, retain_graph=True):
        """Updates decoder and discriminator used for reconstruction space.

        Args:
            data (list): List of ndarrays
            retain_graph (bool):

        Returns:
            None
        """
        # Get the output dimension of classifier
        num_classes = self.options["n_domains"]
        # Get the data
        Xdata, z, dlabels = data
        # Concatenate labels to z to use decoder as conditional decoder
        z_cond = th.cat((z, dlabels.float().view(-1, num_classes)), dim=1)
        
        # 1)----  Start of Discriminator update: Autoencoder in evaluation mode
        self.autoencoder.eval()
        # Forward pass on decoder
        with th.no_grad():
            Xrecon = self.autoencoder.decoder(z_cond)
        # Concatenate labels of image data (repeated 10 times) to its corresponding embedding (i.e. conditional)
        real = th.cat((Xdata, dlabels.float().view(-1, num_classes)), dim=1)
        # Concatenate domain labels of data to its corresponding reconstructions (to make it conditional)
        fake = th.cat((Xrecon, dlabels.float().view(-1, num_classes)), dim=1)
        # Get predictions for real samples
        pred_fake = self.discriminator_x(fake.detach())
        # Get predictions for fake samples
        pred_real = self.discriminator_x(real)
        # Compute discriminator loss
        disc_loss = self.disc_loss(pred_real, pred_fake)
        # Reset optimizer
        self.optimizer_disc_x.zero_grad()
        # Backward pass
        disc_loss.backward(retain_graph=retain_graph)
        # Update parameters of discriminator
        self.optimizer_disc_x.step()
        
        # 2)---- Start of Generator update: Autoencoder in train mode
        self.autoencoder.decoder.train()
        # Discriminator in evaluation mode
        self.discriminator_x.eval()
        #  Forward pass on Autoencoder
        Xrecon = self.autoencoder.decoder(z_cond)
        # Concatenate domain labels of data to its corresponding reconstructions (i.e. conditional)
        fake = th.cat((Xrecon, dlabels.float().view(-1, num_classes)), dim=1)
        # Get predictions for real samples
        pred_fake = self.discriminator_x(fake)
        # Compute discriminator loss
        gen_loss = self.gen_loss(pred_fake)
        # Reset optimizer
        self.optimizer_gen_x.zero_grad()
        # Backward pass
        gen_loss.backward(retain_graph=retain_graph)
        # Update parameters of discriminator
        self.optimizer_gen_x.step()
        # Turn training mode back on. Default mode is training
        self.set_mode()
        # Record losses
        self.loss["aae_loss_x"].append([disc_loss.item(), gen_loss.item()])
        # Delete losses (graphs) for efficient memory usage
        self.clean_up_memory([disc_loss, gen_loss])

    def shuffle_tensors(self, data_list):
        """Shuffles rows of tensors

        Args:
            data_list (list): List of tensors

        Returns:
            tensor:
        """
        # Shuffle input and domain labels to precent clf from learning a trivial solution.
        random_indexes = th.randperm(3*self.options["batch_size"])
        # Shuffled data
        data_shuffled = [data[random_indexes, :] for data in data_list]
        # Return
        return data_shuffled

    def clean_up_memory(self, losses):
        """Deletes losses with attached graph, and cleans up memory"""
        for loss in losses: del loss
        gc.collect()

    def process_batch(self, ds1_dict, ds2_dict, ds3_dict):
        """Concatenates arrays from different data sources into one, and pushes it to the device"""
        # Process the batch i.e. turning it into a tensor
        Xds1, Xds2, Xds3 = [d['tensor'].to(self.device) for d in [ds1_dict, ds2_dict, ds3_dict]]
        # Get labels
        Yds1, Yds2, Yds3 = [d['binary_label'].to(self.device) for d in [ds1_dict, ds2_dict, ds3_dict]]
        # Concatenate data from different sources
        Xdata, labels = th.cat((Xds1, Xds2, Xds3), dim=0), th.cat((Yds1, Yds2, Yds3), dim=0)
        # Get domain labels
        dlabels = self.domain_labels
        # Return
        return Xdata, labels, dlabels

    def update_log(self, epoch, batch):
        """Updates the messages displayed during training and evaluation"""
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch - 1}], Batch:[{batch}], Recon. loss:{self.loss['rloss_b'][-1]:.4f}"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch - 1}] training loss:{self.loss['rloss_e'][-1]:.4f}, val loss:{self.loss['vloss_e'][-1]:.4f}"
        
        # Add generator and discriminator losses
        if self.options["adv_training"]:
            description += f", Disc-Z loss:{self.loss['aae_loss_z'][-1][0]:.4f}, Gen-Z:{self.loss['aae_loss_z'][-1][1]:.4f}"
            description += f", Disc-X loss:{self.loss['aae_loss_x'][-1][0]:.4f}, Gen-X:{self.loss['aae_loss_x'][-1][1]:.4f}"
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        """Sets the mode of the models, either as .train(), or .eval()"""
        for _, model in self.model_dict.items():
            model.train() if mode == "training" else model.eval()

    def save_weights(self):
        """Used to save weights."""
        for model_name in self.model_dict:
            th.save(self.model_dict[model_name], self._model_path + "/" + model_name + ".pt")
        print("Done with saving models.")

    def load_models(self):
        """Used to load weights saved at the end of the training."""
        for model_name in self.model_dict:
            model = th.load(self._model_path + "/" + model_name + ".pt", map_location=self.device)
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def print_model_summary(self):
        """Displays model architectures as a sanity check to see if the models are constructed correctly."""
        # Summary of the model
        description = f"{40 * '-'}Summary of the models (an Autoencoder and two Discriminators):{40 * '-'}\n"
        description += f"{34 * '='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34 * '='}\n"
        description += f"{self.autoencoder}\n"
        # Summary of Discriminator
        if self.options["adv_training"]:
            description += f"{30 * '='} Discriminator for latent {30 * '='}\n"
            description += f"{self.discriminator_z}\n"
            description += f"{30 * '='} Discriminator for reconstruction {30 * '='}\n"
            description += f"{self.discriminator_x}\n"
        # Print model architecture
        print(description)

    def _update_model(self, loss, optimizer, retain_graph=True):
        """Does backprop, and updates the model parameters

        Args:
            loss ():
            optimizer ():
            retain_graph (bool):

        Returns:
            None
        """
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def _set_scheduler(self):
        """Sets a scheduler for learning rate of autoencoder"""
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_ae, step_size=1, gamma=0.97)

    def _set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = self.options["paths"]["results"]
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", self.options["model_mode"], "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", self.options["model_mode"], "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", self.options["model_mode"], "loss")

    def _adam(self, params, lr=1e-4):
        """Sets up Adam optimizer using model params"""
        return th.optim.Adam(itertools.chain(*params), lr=lr, betas=(0.9, 0.999))

    def _tensor(self, data):
        """Turns numpy arrays to torch tensors"""
        return th.from_numpy(data).to(self.device).float()
    
    def one_hot_embedding(self, labels, num_classes):
        """Converts labels to one-hot encoded form.

        Args:
          labels (LongTensor):  class labels, sized [N,].
          num_classes (int):  number of classes.

        Returns:
          None
        """
        y = th.eye(num_classes) 
        return y[labels].to(self.device)

    def set_domain_labels(self):
        # Assign each domain to a label: domain-1:0, domain-2:1 and so on.
        self.domain_labels = []
        # i = number that the domain is assigned to.
        for i in range(self.options["n_domains"]):
            # Repeat each for number of batch size so that we have label for each data point from each domain
            self.domain_labels += self.options["batch_size"] * [i]
        # Turn labels to torch tensor
        self.domain_labels = th.from_numpy(np.array(self.domain_labels))
        # Turn them into one-hot embeddings, shape: (3 x batch_size, number of domains)
        self.domain_labels = self.one_hot_embedding(self.domain_labels, self.options["n_domains"])

    def supervised_gaussian_mixture(self, label_indices):
        """Samples data from the GMM prior

        Args:
            label_indices (ndarray): 1D array of cluster/class labels

        Returns:
            ndarray, ndarray: 2D and 1D numpy arrays
        """
        batchsize = self.options["n_domains"]*self.options["batch_size"]
        ndim = self.options["dims"][-1]
        num_clabels = self.options["n_cohorts"]

        if ndim % 2 != 0:
            raise Exception("ndim must be a multiple of 2.")

        x_var = 0.5
        y_var = 0.5
        x = np.random.normal(0, x_var, (batchsize, ndim // 2))
        y = np.random.normal(0, y_var, (batchsize, ndim // 2))
        z = np.empty((batchsize, ndim), dtype=np.float32)
        for batch in range(batchsize):
            for zi in range(ndim // 2):
                z[batch, zi * 2:zi * 2 + 2] = self.gm_sample(x[batch, zi], y[batch, zi], label_indices[batch],
                                                             num_clabels)
        return z, label_indices.cpu().numpy()

    def gm_sample(self, x, y, label, num_clabels):
        """

        Args:
            x (ndarray): 1D array of float numbers
            y (ndarray): 1D array of float numbers
            label (ndarray): 1D array of cluster labels
            num_clabels (int): Number of clusters/classes in a domain

        Returns:
            ndarray: 2D numpy array
        """
        shift = 1.4
        r = 2.0 * np.pi / float(num_clabels) * float(label)
        new_x = x * np.cos(r) - y * np.sin(r)
        new_y = x * np.sin(r) + y * np.cos(r)
        new_x += shift * np.cos(r)
        new_y += shift * np.sin(r)
        return np.array([new_x, new_y]).reshape((2,))