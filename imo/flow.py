#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

from nflows import transforms, flows
from nflows import distributions as distributions_
from torch import nn, Tensor, optim
from sbi.utils.sbiutils import standardizing_transform

from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralDensityEstimator(object):
    """
    Neural density estimator class. Basically a wrapper.
    """

    def __init__(
        self,
        normalize: bool = True,
        initial_pos: dict = None,
        method: str = "nsf",
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_bins: int = 10,
        embedding_net: nn.Module = nn.Identity(),
        **kwargs,
    ):
        """
        Initialize neural density estimator.
        Parameters
        ----------
        normalize: bool.
            Whether to z-score the data that you want to model.
        initial_pos: dict.
            Initial position of the density,
            e.g., ``{'bounds': [[1, 2], [0, 1]], 'std': [1, .05]}``.
            It includes the bounds for sampling the means of Gaussians,
            and the standard deviations of the Gaussians.
        method: str.
            Method to use for density estimation, either ``'nsf'`` or ``'maf'``.
        hidden_features: int.
            Number of hidden features.
        num_transforms: int.
            Number of transforms.
        num_bins: int.
            Number of bins used for the splines.
        embedding_net: torch.nn.Module.
            Optional embedding network for y.
        kwargs: dict.
            Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert method in ["nsf", "maf"], "Method must be either 'nsf' or 'maf'."
        self.method = method

        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.num_bins = num_bins  # only works for NSF
        self.normalize = normalize

        if initial_pos is None:
            raise ValueError(
                "initial_pos must be specified. Please see the documentation."
            )
        assert len(initial_pos["bounds"]) == len(
            initial_pos["std"]
        ), "The length of bounds and std must be the same."
        self.initial_pos = initial_pos

        self.embedding_net = embedding_net
        self.train_loss_history = []
        self.valid_loss_history = []

    def build(self, batch_theta: Tensor, optimizer: str = "adam", lr=1e-3, **kwargs):
        """
        Build the neural density estimator based on input data.
        Parameters
        ----------
        batch_theta: torch.Tensor.
            The input data whose distribution will be modeled by NDE.
        optimizer: float.
            The optimizer to use for training, default is ``Adam``.
        lr: float.
            The learning rate for the optimizer.
        """
        if not torch.is_tensor(batch_theta):
            batch_theta = torch.tensor(batch_theta, device=self.device)
        self.batch_theta = batch_theta

        self.net = build_maf(
            batch_x=batch_theta,
            z_score_x=self.normalize,
            initial_pos=self.initial_pos,
            hidden_features=self.hidden_features,
            num_transforms=self.num_transforms,
            embedding_net=self.embedding_net,
            device=self.device,
            **kwargs,
        )
        self.net.to(self.device)

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer {optimizer}, only support 'Adam' now.")

    def _train(self, n_epochs: int = 2000, suffix: str = "nde"):
        """
        Train the neural density estimator based on input data.
        Here we use the ``log(P)`` loss. This function is not used in the ``popsed`` project.
        Parameters
        ----------
        n_epochs: int.
            Number of epochs to train.
        display: bool.
            Whether to display the training loss.
        suffix: str.
            Suffix to add to the output file.
        """
        min_loss = -19
        patience = 5
        self.best_loss_epoch = 0
        self.net.train()

        for epoch in trange(n_epochs, desc="Training NDE", unit="epochs"):
            self.optimizer.zero_grad()
            loss = -self.net.log_prob(self.batch_x).mean()
            loss.backward()
            self.optimizer.step()
            self.train_loss_history.append(loss.item())

            if loss.item() < min_loss:
                min_loss = loss.item()
                if epoch - self.best_loss_epoch > patience:
                    # Don't save model too frequently
                    self.best_loss_epoch = epoch
                    self.save_model(f"best_loss_model_{suffix}_{self.method}.pkl")

        if min_loss == -18:
            raise Warning("The training might be failed, try more epochs")

    def sample(self, n_samples: int = 1000):
        """
        Sample according to the fitted NDE.
        Parameters
        ----------
        n_samples: int.
            Number of samples to draw.
            If the number is too large, the GPU memory may be insufficient.
        Returns
        -------
        samples: torch.Tensor.
            Samples drawn from the NDE.
        """
        return self.net.sample(n_samples)

    def save_model(self, filename):
        """
        Save NDE model.
        Parameters
        ----------
        filename: str.
            Name of the file to save the model.
        """
        torch.save(self, filename)
        # with open(filename, 'wb') as f:
        #    pickle.dump(self, f)


def build_maf(
    batch_x: Tensor = None,
    z_score_x: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    # device: str = 'cuda',
    initial_pos: dict = {"bounds": [[1, 2], [0, 1]], "std": [1, 0.05]},
    **kwargs,
):
    """Builds MAF to describe p(x).
    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, i.e., whether do normalization.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()

    if x_numel == 1:
        raise Warning(
            f"In one-dimensional output space, this flow is limited to Gaussians"
        )

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=x_numel,
                        hidden_features=hidden_features,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=torch.tanh,
                        dropout_probability=0.0,
                        use_batch_norm=False,
                    ),
                    transforms.RandomPermutation(features=x_numel),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    if z_score_x:
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    if initial_pos is not None:
        _mean = np.random.uniform(
            low=np.array(initial_pos["bounds"])[:, 0],
            high=np.array(initial_pos["bounds"])[:, 1],
        )
        transform_init = transforms.AffineTransform(
            shift=torch.Tensor(-_mean) / torch.Tensor(initial_pos["std"]),
            scale=1.0 / torch.Tensor(initial_pos["std"]),
        )
        transform = transforms.CompositeTransform([transform_init, transform])

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net).to(device)

    return neural_net
