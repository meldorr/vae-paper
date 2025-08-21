import torch.nn as nn
from encoder import Encoder
from decoder import TCDecoder
from lsr import NormalLSR, VampPriorLSR
import torch
import torch.nn.functional as F
from torch.distributions.distribution import Distribution

class TCVAE(nn.Module):

    def __init__(
            self,
            hparams,
            inputdim = 4,
            seq_length = 200
    ):
        
        super().__init__()
        
        self.pseudo_gamma = 0.1
        self.hparams = hparams
        self.inputdim = inputdim
        self.seq_length = seq_length
        self.scale = nn.Parameter(
            torch.Tensor([self.hparams["scale"]]), requires_grad=True
        )

        self.encoder = Encoder(
            input_dim=self.inputdim,
            out_dim=hparams["h_dims"][-1],
            h_dims=hparams["h_dims"][:-1],
            kernel_size=hparams["kernel_size"],
            dilation_base=hparams["dilation_base"],
            sampling_factor=hparams["sampling_factor"],
            h_activ=nn.ReLU(),
            dropout=self.hparams["dropout"]
        )

        self.h_dim = self.hparams["h_dims"][-1] * (
            int(self.seq_length / self.hparams["sampling_factor"])
        )

        if self.hparams["prior"] == 'standard':
            self.lsr = NormalLSR(
                input_dim=self.h_dim,
                out_dim=self.hparams["encoding_dim"]
            )
        elif self.hparams["prior"] == 'vampprior':
            self.lsr = VampPriorLSR(
                original_dim=self.inputdim,
                original_seq_len=self.seq_length,
                input_dim=self.h_dim,
                out_dim=self.hparams["encoding_dim"],
                encoder=self.encoder,
                n_components=self.hparams["n_components"]
            )
        elif self.hparams["prior"] == 'exemplar':
            print("unavailable now, and will use NormalLSR instead")
            self.lsr = NormalLSR(
                input_dim=self.h_dim,
                out_dim=self.hparams["encoding_dim"]
            )
        else:
            raise ValueError(f"Unknown LSR type: {self.hparams.prior}")

        self.decoder = TCDecoder(
            input_dim=self.hparams["encoding_dim"],
            out_dim=self.inputdim,
            h_dims=self.hparams["h_dims"][::-1],
            seq_len=self.seq_length,
            kernel_size=self.hparams["kernel_size"],
            dilation_base=self.hparams["dilation_base"],
            sampling_factor=self.hparams["sampling_factor"],
            dropout=self.hparams["dropout"],
            h_activ=nn.ReLU()
        )

        self.out_activ = nn.Identity()

    def forward(self, x):
        h = self.encoder(x)
        # When batched, q is a collection of normal posterior
        q = self.lsr(h)
        z = q.rsample()
        # decode z
        x_hat = self.out_activ(self.decoder(z))
        return self.lsr.dist_params(q), z, x_hat

    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the gaussian likelihood and the Kullback-Leibler divergence
        to get the ELBO loss function.

        .. math::

            \\mathcal{L}_{ELBO} = \\alpha \\times KL(q(z|x) || p(z))
            - \\beta \\times \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
        """
        x, _ = batch
        dist_params, z, x_hat = self.forward(x)

        # std of decoder distribution (init at 1)
        # self.scale = nn.Parameter(
        #     torch.Tensor([torch.sqrt(F.mse_loss(x, x_hat))]),
        #     requires_grad=False,
        # )
        # gamma = self.scale

        # Regular VAE LOSS
        # log likelihood loss (reconstruction loss)
        llv_loss = -self.gaussian_likelihood(x, x_hat)
        llv_coef = self.hparams["llv_coef"]
        # kullback-leibler divergence (regularization loss)
        q_zx = self.lsr.get_posterior(dist_params)
        p_z = self.lsr.get_prior()
        kld_loss = self.kl_divergence(z, q_zx, p_z)
        kld_coef = self.hparams["kld_coef"]

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = kld_coef * kld_loss + llv_coef * llv_loss
        elbo = elbo.mean()

        # Regularization to make the pseudo-inputs close to their reconstruction
        if self.hparams["reg_pseudo"]:
            # Calculate pseudo-inputs for regularization term
            pseudo_X = self.lsr.pseudo_inputs_NN(self.lsr.idle_input)
            pseudo_X = pseudo_X.view(
                (pseudo_X.shape[0], x.shape[1], x.shape[2])
            )
            pseudo_dist_params, pseudo_z, pseudo_x_hat = self.forward(pseudo_X)

            # Regularization term for pseudo_inputs
            # log likelihood loss (reconstruction loss)
            pseudo_llv_loss = -self.gaussian_likelihood(pseudo_X, pseudo_x_hat)
            # kullback-leibler divergence (regularization loss)
            pseudo_q_zx = self.lsr.get_posterior(pseudo_dist_params)
            pseudo_kld_loss = self.kl_divergence(pseudo_z, pseudo_q_zx, p_z)

            pseudo_elbo = (
                kld_coef * pseudo_kld_loss + llv_coef * pseudo_llv_loss
            )
            pseudo_elbo = (x.shape[0] / pseudo_X.shape[0]) * pseudo_elbo.mean()
            elbo = elbo + self.pseudo_gamma * pseudo_elbo

        # ELBO loss from Diagnosing and Enhancing VAE Models
        # Values are very close, but we can have access to the gamma parameter
        # kl = (
        #     torch.sum(self.kl_loss(dist_params[1], dist_params[2])) / batch_size
        # )
        # gen = torch.sum(self.gen_loss(x, x_hat, gamma)) / batch_size
        # elbo = kl + gen

        # self.log_dict(
        #     {
        #         "train_loss": elbo,
        #         "kl_loss": kld_loss.mean(),
        #         "recon_loss": llv_loss.mean(),
        #     }
        # )
        return elbo, kld_loss.mean(), llv_loss.mean()

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, info = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, x_hat, info
    
    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor):
        """Computes the gaussian likelihood.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): mean decoded from :math:`z`.

        .. math::

            \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
            \\text{ with } p(.|z_{i})
            \\sim \\mathcal{N}(\\hat{x_{i}},\\,\\sigma^{2})

        .. note::
            The scale :math:`\\sigma` can be defined in config and will be
            accessible with ``self.scale``.
        """
        mean = x_hat
        dist = torch.distributions.Normal(mean, self.scale)
        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(
        self, z: torch.Tensor, p: Distribution, q: Distribution
    ) -> torch.Tensor:
        """Computes Kullback-Leibler divergence :math:`KL(p || q)` between two
        distributions, using Monte Carlo Sampling.
        Evaluate every z of the batch in its corresponding posterior (1st z with 1st post, etc..)
        and every z in the prior

        Args:
            z (torch.Tensor): A sample from p (the posterior).
            p (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the posetrior)
            q (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the prior)

        Returns:
            torch.Tensor: A batch of KL divergences of shape `z.size(0)`.

        .. note::
            Make sure that the `log_prob()` method of both Distribution
            objects returns a 1D-tensor with the size of `z` batch size.
        """
        log_p = p.log_prob(z)
        log_q = q.log_prob(z)
        return log_p - log_q

    # The 2 terms here are part of the other formulation of the loss
    # present in Diagnosing and Enhancing VAE Models
    def gen_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, gamma: torch.Tensor
    ):
        """Computes generation loss in TwoStages VAE Model
        Args :
            x : input data
            x_hat : reconstructed data
            gamma : decoder std (scalar as every distribution in the decoder has the same std)

        To use it within the learning : take the sum and divide by the batch size
        """
        HALF_LOG_TWO_PI = 0.91893

        loggamma = torch.log(gamma)
        return (
            torch.square((x - x_hat) / gamma) / 2.0 + loggamma + HALF_LOG_TWO_PI
        )

    def kl_loss(self, mu: torch.Tensor, std: torch.Tensor):
        """Computes close form of KL for gaussian distributions
        Args :
            mu : encoder means
            std : encoder stds

        To use it within the learning : take the sum and divide by the batch size
        """
        logstd = torch.log(std)
        return (torch.square(mu) + torch.square(std) - 2 * logstd - 1) / 2.0
