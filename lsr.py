from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Independent, MixtureSameFamily, MultivariateNormal, Normal
)
from torch.distributions.categorical import Categorical


class NormalLSR(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, fix_prior: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fix_prior = fix_prior


        self.z_loc = nn.Linear(input_dim, out_dim)
        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, out_dim))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        self.out_dim = out_dim
        self.dist = Normal

        self.prior_loc = nn.Parameter(
            torch.zeros((1, out_dim)), requires_grad=False
        )
        self.prior_log_var = nn.Parameter(
            torch.zeros((1, out_dim)), requires_grad=False
        )
        self.register_parameter("prior_loc", self.prior_loc)
        self.register_parameter("prior_log_var", self.prior_log_var)

    def forward(self, hidden) -> Distribution:
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        return Independent(self.dist(loc, (log_var / 2).exp()), 1)

    def dist_params(self, p: Independent) -> List[torch.Tensor]:
        return [p.base_dist.loc, p.base_dist.scale]

    def get_posterior(self, dist_params: List[torch.Tensor]) -> Independent:
        return Independent(self.dist(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> Independent:
        return Independent(
            self.dist(self.prior_loc, (self.prior_log_var / 2).exp()), 1
        )


class VampPriorLSR(nn.Module):
    """VampPrior Latent Space Regularization. https://arxiv.org/pdf/1705.07120.pdf

    Args:
        original_dim(int): number of features for each trajectory (usually 4)
        original_seq_len(int): sequence length of one trajectory (usually 200)
        input_dim (int): size of each input sample after the encoder NN
        out_dim (int):size of each output sample, dimension of the latent distributions
        encoder (nn.Module) : Neural net used for the encoder
        n_components (int, optional): Number of components in the Gaussian
            Mixture of the VampPrior. Defaults to ``500``.
    """

    def __init__(
        self,
        original_dim: int,
        original_seq_len: int,
        input_dim: int,
        out_dim: int,
        encoder: nn.Module,
        n_components: int,
        fix_prior: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fix_prior = fix_prior


        self.original_dim = original_dim
        self.seq_len = original_seq_len
        self.encoder = encoder
        self.n_components = n_components

        # We don't use customMSF here because we don't need to chose one component of the prior when sampling
        self.dist = MixtureSameFamily
        self.comp = Normal
        self.mix = Categorical

        # Posterior Parameters -> those need to be only gaussian in the paper
        z_loc_layers = []
        z_loc_layers.append(nn.Linear(input_dim, out_dim))
        self.z_loc = nn.Sequential(*z_loc_layers)

        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, out_dim))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        # prior parameters
        # Input to the NN that will produce the pseudo inputs
        self.idle_input = torch.autograd.Variable(
            torch.eye(n_components, n_components), requires_grad=False
        )

        # NN that transform the idle_inputs into the pseudo_inputs that will be transformed
        # by the encoder into the means of the VampPrior
        pseudo_inputs_layers = []
        pseudo_inputs_layers.append(nn.Linear(n_components, n_components))
        pseudo_inputs_layers.append(nn.ReLU())
        pseudo_inputs_layers.append(
            nn.Linear(
                n_components,
                (original_dim * original_seq_len),
            )
        )
        pseudo_inputs_layers.append(nn.Hardtanh(min_val=-1.0, max_val=1.0))
        self.pseudo_inputs_NN = nn.Sequential(*pseudo_inputs_layers)

        # decouple variances of posterior and prior componenents
        prior_log_var_layers = []
        prior_log_var_layers.append(nn.Linear(input_dim, out_dim))
        prior_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.prior_log_var_NN = nn.Sequential(*z_log_var_layers)

        # In Vamprior, the weights of the GM are all equal
        # Here they are trained
        self.prior_weights = nn.Parameter(
            torch.ones((1, n_components)), requires_grad=True
        )

        self.register_parameter("prior_weights", self.prior_weights)

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """[summary]

        Args:
            hidden (torch.Tensor): output of encoder

        Returns:
            Distribution: corresponding posterior distribution
        """

        # Calculate the posterior parameters :
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        scales = (log_var / 2).exp()

        # calculate the prior paramters :
        X = self.pseudo_inputs_NN(self.idle_input)
        X = X.view((X.shape[0], self.original_dim, self.seq_len))
        pseudo_h = self.encoder(X)
        self.prior_means = self.z_loc(pseudo_h)
        # self.prior_log_vars = self.z_log_var(pseudo_h)
        self.prior_log_vars = self.prior_log_var_NN(pseudo_h)

        # return the posterior : a single multivariate normal
        return Independent(self.comp(loc, scales), 1)

    # Only for the posterior distribution
    def dist_params(self, p: MixtureSameFamily) -> Tuple:
        return [p.base_dist.loc, p.base_dist.scale]

    # Is a signle multivariate normal
    def get_posterior(self, dist_params: Tuple) -> Distribution:
        return Independent(self.comp(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> MixtureSameFamily:
        return self.dist(
            self.mix(logits=self.prior_weights.view(self.n_components)),
            Independent(
                self.comp(
                    self.prior_means,
                    (self.prior_log_vars / 2).exp(),
                ),
                1,
            ),
        )
