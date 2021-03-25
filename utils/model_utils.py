"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Library of models and related support functions.
Models: Autoencoder, CNN-Encoder, CNN-Decoder, FC-Encoder, FC-Decoder, Discriminator, Classifier

Dictionary:
FC = Fully-connected
"""

import os
import copy
import numpy as np
import pandas as pd
import torch as th
from torch import nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, options):
        super(Autoencoder, self).__init__()
        self.options = options
        self.encoder = CNNEncoder(options) if options["convolution"] else Encoder(options)
        self.decoder = CNNDecoder(options) if options["convolution"] else Decoder(options)

    def forward(self, data):
        # Get input and corresponding domain labels
        x, y = data if self.options["conditional"] else [data, data]
        # Forward pass on Encoder
        mean, logvar = self.encoder(x)
        # Compute latent by sampling if the model is VAE, or else the latent is just the mean.
        latent = sampling([mean, logvar, self.options]) if self.options["model_mode"] in ["vae", "bvae"] else mean
        # Do L2 normalization
        latent = F.normalize(latent, p=self.options["p_norm"], dim=1) if self.options["normalize"] else latent
        # Concatenate latent with domain labels since Decoder is conditional
        latent_conditional = th.cat((latent, y), dim=1) if self.options["conditional"] else latent
        # Forward pass on decoder
        x_recon = self.decoder(latent_conditional)
        return x_recon, latent, mean, logvar


class CNNEncoder(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: (mean, logvar) if in VAE mode. Else it return (z, z).

    Encoder model.
    """

    def __init__(self, options):
        super(CNNEncoder, self).__init__()
        # Container to hold layers of the architecture in order
        self.layers = nn.ModuleList()
        # Get configuration that contains architecture and hyper-parameters
        self.options = copy.deepcopy(options)
        # Get the dimensions of all layers
        dims = options["conv_dims"]
        # Input image size. Example: 28 for a 28x28 image.
        img_size = self.options["img_size"]
        # Get dimensions for convolution layers in the following format: [i, o, k, s, p, d]
        # i=input channel, o=output channel, k = kernel size, s = stride, p = padding, d = dilation
        convolution_layers = dims[:-1]
        # Final output dimension of encoder i.e. dimension of projection head
        output_dim = dims[-1]
        # Go through convolutional layers
        for layer_dims in convolution_layers:
            i, o, k, s, p, d = layer_dims
            self.layers.append(nn.Conv2d(i, o, k, stride=s, padding=p, dilation=d))
            # BatchNorm if True
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm2d(o))
            # Add activation
            self.layers.append(nn.LeakyReLU(inplace=False))
            # Dropout if True
            if options["isDropout"]:
                self.layers.append(nn.Dropout2d(options["dropout_rate"]))
        # Do global average pooling over spatial dimensions to make Encoder agnostic to input image size
        self.global_ave_pool = global_ave_pool
        # First linear layer, which will be followed with non-linear activation function in the forward()
        # self.linear_layer1 = nn.Linear(o, o)
        # Mean layer for final projection
        self.mean = nn.Linear(o, output_dim)
        # Logvar layer for final projection
        self.logvar = nn.Linear(o, output_dim)

    def forward(self, x):
        # batch size, height, width, channel of the input
        bs, h, w, ch = x.size()
        # Forward pass on convolutional layers
        for layer in self.layers:
            x = layer(x)
        # Global average pooling over spatial dimensions. This is also used as learned representation.
        h = self.global_ave_pool(x)
        # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
        z = h  # F.relu(self.linear_layer1(h))
        # Apply linear layer for mean
        mean = self.mean(z)
        # Create a placeholder for logvar, which is redundant by default
        logvar = mean
        # If the model is Variational Autoencoder, compute logvar.
        logvar = self.logvar(z) if self.options["model_mode"] in ["vae", "bvae"] else logvar
        return mean, logvar


class CNNDecoder(nn.Module):
    def __init__(self, options):
        super(CNNDecoder, self).__init__()
        # Container to hold layers of the architecture in order
        self.layers = nn.ModuleList()
        # Get configuration that contains architecture and hyper-parameters
        self.options = copy.deepcopy(options)
        # Get the dimensions of all layers by reversing the order of convolutional layers
        dims = self.options["conv_dims"][::-1]
        # Get dimensions for convolution layers in the following format: [i, o, k, s, p, d]
        # i=input channel, o=output channel, k = kernel size, s = stride, p = padding, d = dilation
        self.convolution_layers = dims[1:]
        # Input dimension of decoder. Add number-of-domains to the input dimension of Decoder to make it conditional
        input_dim = dims[0] + self.options["n_domains"]
        # Dimensions for first deconvolutional layer = dimensions of last convolutional layer
        o, i, k, s, p, d = self.convolution_layers[0]
        # Starting image size
        self.img_size = 4
        # First linear layer with shape (bottleneck dimension, output channel size of last conv layer in CNNEncoder)
        self.first_layer = nn.Linear(input_dim, i * self.img_size * self.img_size)
        # Add deconvolutional layers
        for layer_dims in self.convolution_layers:
            # Get dimensions. Note that i, o are swapped since we are down sampling channels in Decoder
            o, i, k, s, p, d = layer_dims
            self.layers.append(nn.ConvTranspose2d(i, o, k, stride=s, padding=p, dilation=d))
            # BatchNorm if True
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm2d(o))
            # Add activation
            self.layers.append(nn.LeakyReLU(inplace=False))
            # Dropout if True
            if options["isDropout"]:
                self.layers.append(nn.Dropout2d(options["dropout_rate"]))
        # Sigmoid function to get probabilities
        self.probs = nn.Sigmoid()

    def forward(self, z):
        # batch size, latent dimension of the input
        bs, latent_dim = z.size()
        # Dimensions for first deconvolutional layer. ote that i, o are swapped (compared to CNNEncoder).
        o, i, k, s, p, d = self.convolution_layers[0]
        h = self.first_layer(z)
        h = h.view(bs, i, self.img_size, self.img_size)
        # Forward pass on convolutional hidden layers
        for layer in self.layers:
            h = layer(h)
        # Apply final linear layer to reduce dimension
        probs = self.probs(h)
        return probs


class Encoder(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: (mean, logvar) if in VAE mode. Else it return (z, z).

    Encoder model.
    """

    def __init__(self, options):
        super(Encoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # Forward pass on hidden layers
        self.hidden_layers = HiddenLayers(self.options, network="encoder")
        # Compute the mean i.e. bottleneck in Autoencoder
        self.mean = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])
        # if the model is Variational Autoencoder, compute logvar.
        self.logvar = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute the mean i.e. bottleneck in Autoencoder
        mean = self.mean(h)
        # Create a placeholder for logvar, which is redundant by default
        logvar = mean
        # if the model is Variational Autoencoder, compute logvar.
        logvar = self.logvar(h) if self.options["model_mode"] in ["vae", "bvae"] else logvar
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # Revert the order of hidden units so that we can build a Decoder, which is the symmetric of Encoder
        self.options["dims"] = self.options["dims"][::-1]
        # Add number-of-domains to the input dimension of Decoder to make it conditional
        self.options["dims"][0] = self.options["dims"][0] + self.options["n_domains"]
        # Add hidden layers
        self.hidden_layers = HiddenLayers(self.options, network="decoder")
        # Compute logits and probabilities
        self.logits = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])
        self.probs = nn.Sigmoid()

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute logits
        logits = self.logits(h)
        # Compute probabilities
        #         probs = self.probs(logits)
        return logits


class Classifier(nn.Module):
    def __init__(self, options, input_dim=None):
        super(Classifier, self).__init__()
        self.options = copy.deepcopy(options)
        # Define the input dimension of Classifier - By default, it is same as the latent dim of Autoencoder
        latent_dim = self.options["conv_dims"][-1] if options["convolution"] else self.options["dims"][-1]
        # If input_dim is defined, use it as input dimension of Discriminator
        input_dim = input_dim or latent_dim
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, options["output_dim"])
        self.probs = nn.Sigmoid()

    def forward(self, h):
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        logits = self.logits(h)
        probs = self.probs(logits)
        return probs


class Discriminator(nn.Module):
    def __init__(self, options, input_dim=None):
        super(Discriminator, self).__init__()
        # Assign a copy of options to self
        self.options = copy.deepcopy(options)
        # Define the input dimension of Discriminator - By default, it is same as the latent dim of Autoencoder
        latent_dim = self.options["conv_dims"][-1] if options["convolution"] else self.options["dims"][-1]
        latent_dim = latent_dim + self.options["n_cohorts"]
        # If input_dim is defined, use it as input dimension of Discriminator
        input_dim = input_dim or latent_dim
        # Define hidden layers
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, 1)
        self.probs = nn.Sigmoid()

    def forward(self, h):
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        logits = self.logits(h)
        probs = self.probs(logits)
        return probs


class HiddenLayers(nn.Module):
    def __init__(self, options, network="encoder"):
        super(HiddenLayers, self).__init__()
        self.layers = nn.ModuleList()
        dims = options["dims"]

        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm1d(dims[i]))

            self.layers.append(nn.LeakyReLU(inplace=False))
            if options["isDropout"]:
                self.layers.append(nn.Dropout(options["dropout_rate"]))

    def forward(self, x):
        for layer in self.layers:
            # You could do an if isinstance(layer, nn.Type1) maybe to check for types
            x = layer(x)

        return x


class Flatten(nn.Module):
    "Flattens tensor to 2D: (batch_size, feature dim)"

    def forward(self, x):
        return x.view(x.shape[0], -1)


def global_ave_pool(x):
    """Global Average pooling of convolutional layers over the spatioal dimensions.
    Results in 2D tensor with dimension: (batch_size, number of channels) """
    return th.mean(x, dim=[2, 3])


def compute_image_size(args):
    """Computes resulting image size after a convolutional layer
    i=input channel, o=output channel, k = kernel size, s = stride, p = padding, d = dilation
    old_size = size of input image,
    new_size= size of output image.
    """
    old_size, i, o, k, s, p, d = args
    new_size = int((old_size + 2 * p - d * (k - 1) - 1) // s) + 1
    return new_size


def sampling(args):
    """
    :param list of args: The mean and log-variance parameters of Gaussian distribution.
    :return: Samples from latent layer.
    """
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    mu, logvar, options = args
    stdev = options["stddev"]

    # Re-parameterization trick
    eps = th.normal(mean=0, std=stdev, size=logvar.size())
    eps = eps.to(device).float()
    return mu + eps * th.exp(0.5 * logvar)


class AddGaussNoise(object):
    def __init__(self, options, mean=0, std=1.0):
        self.std = std
        self.mean = mean
        self.device = th.device(options["device"])

    def __call__(self, tensor):
        rnoise = th.randn(tensor.size()).to(self.device).float()
        return tensor + rnoise * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)
