"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Library of loss functions.
"""

import numpy as np
import torch as th



def get_th_vae_loss(recon_loss, mu, logvar, options):
    """
    :param recon_loss:
    :param mu:
    :param logvar:
    :param options:
    :return:
    """
    kl_loss = 0

    if options["model_mode"] in ["vae", "bvae"]:
        # This does not work although this is how it should be:
        # KL Divergence computed through re-parameterization
        # kl_loss = 1 + logvar - th.square(mu) - th.exp(logvar)
        # kl_loss = th.sum(kl_loss) #th.sum(kl_loss)
        # kl_loss *= -0.5

        # This works since we sum across dimensions, and take sample mean. This works with reconstruction term,
        # where we sum across all dimensions, and use sample mean as well.
        kl_loss = 1 + logvar - th.square(mu) - th.exp(logvar)
        kl_loss = th.sum(kl_loss, dim=-1) #th.sum(kl_loss)  # Try mean() since torch MSELoss is computed using mean over sum (sample size+dimension)
        kl_loss *= -0.5
        kl_loss = th.mean(kl_loss)
        # kl_loss = 1 + logvar - th.square(mu) - th.exp(logvar)
        # kl_loss = th.sum(kl_loss) #th.sum(kl_loss)  # Try mean() since torch MSELoss is computed using mean over sum (sample size+dimension)
        # kl_loss *= -0.5
        # kl_loss = kl_loss
    else:
        # Assume that there is no KL component.
        kl_loss = 0

    beta_kl_loss = options["beta"]*kl_loss
    return recon_loss + beta_kl_loss, kl_loss

def get_generator_loss(fake):
    return -th.mean(th.log(fake+1e-8))

def get_discriminator_loss(real, fake):
    return -th.mean(th.log(real+1e-8) + th.log(1-fake+1e-8))

def getKL(args):
    mu, logvar = args
    kl_loss = 1 + logvar - th.square(mu) - th.exp(logvar)
    kl_loss = th.sum(kl_loss, dim=-1)
    kl_loss *= -0.5
    kl_loss = th.mean(kl_loss)
    return kl_loss


def getKLloss(p, q):
    def kl(target, pred):
        return th.mean(th.sum(target * th.log(target / (pred + 1e-8)), dim=1))
    loss = kl(p, q)
    return loss

def getMSEloss(recon, target):
    dims = list(target.size())
    bs = dims[0]
    loss = th.sum(th.square(recon-target))/bs
    return loss

def getCEloss(recon, target):
    bs, _ = list(target.size())
    loss = F.binary_cross_entropy(recon, target, reduction='sum')/bs
    return loss
