import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        z_dim: int,
        encoder_hid_dims: list,
        decoder_hid_dims: list,
        device: str = "cpu",
    ):
        """
        The basic model for VAE
        Inputs:
            in_dim : [int] dimension of input
            encoder_hid_dims : [list] list of dimension in encoder
            decoder_hid_dims : [list] list of dimension in decoder
            z_dim : [int] dimension of the latent variable
            device : [str] 'cpu' or 'gpu';
        """
        super(VAE, self).__init__()
        setattr(self, "_model_name", "VAE")
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.encoder_hid_dims = encoder_hid_dims
        self.decoder_hid_dims = decoder_hid_dims
        self.device = device

        self.vae_encoder = VAE_Encoder(
            in_dim=self.in_dim,
            hid_dims=self.encoder_hid_dims,
            z_dim=self.z_dim,
            device=self.device,
        )
        self.vae_decoder = VAE_Decoder(
            in_dim=self.in_dim,
            hid_dims=self.decoder_hid_dims,
            z_dim=self.z_dim,
            device=self.device,
        )

    # forward
    def forward(self, x):
        """
        Forward process of VAE
        Inputs:
            x : [tensor] input tensor;
        Outputs:
            recon_x : [tensor] reconstruction of x
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        """
        z, mu, log_var = self.vae_encoder(x.view(x.shape[0], -1))
        recon_x = self.vae_decoder(z)
        return recon_x, mu, log_var

    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        recon_x = self.vae_decoder(z)
        return recon_x


class VAE_Encoder(nn.Module):
    def __init__(self, in_dim: int, hid_dims: list, z_dim: int, device: str = "cpu"):
        super(VAE_Encoder, self).__init__()

        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.z_dim = z_dim
        self.device = device
        self.num_layers = len(self.hid_dims)

        self.fc_encoder = nn.ModuleList()
        self.fc_mu = nn.Linear(self.hid_dims[-1], self.z_dim, device=self.device)
        self.fc_var = nn.Linear(self.hid_dims[-1], self.z_dim, device=self.device)

        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.fc_encoder.append(
                    nn.Linear(self.in_dim, self.hid_dims[layer_index]).to(device)
                )
            else:
                self.fc_encoder.append(
                    nn.Linear(
                        self.hid_dims[layer_index - 1], self.hid_dims[layer_index]
                    ).to(device)
                )

    def reparameterize(self, mu, log_var):
        """
        Gaussian re_parameterization
        Inputs:
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        Outputs:
            z_sample : [tensor] sample from the distribution
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        """
        Forward process of VAE_Encoder
        Inputs:
            x : [tensor] input tensor;
        Outputs:
            z : [tensor] latent variable of x
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        """
        x = x.view(x.shape[0], -1)
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                x = F.relu(self.fc_encoder[layer_index](x))
            else:
                x = F.relu(self.fc_encoder[layer_index](x))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, in_dim: int, hid_dims: list, z_dim: int, device: str = "cpu"):
        super(VAE_Decoder, self).__init__()

        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.z_dim = z_dim
        self.num_layers = len(self.hid_dims)
        self.device = device

        self.fc_decoder = nn.ModuleList()
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.fc_decoder.append(
                    nn.Linear(self.z_dim, self.hid_dims[layer_index]).to(device)
                )
            else:
                self.fc_decoder.append(
                    nn.Linear(
                        self.hid_dims[layer_index - 1], self.hid_dims[layer_index]
                    ).to(device)
                )
        self.fc_decoder.append(nn.Linear(self.hid_dims[-1], self.in_dim).to(device))

    def forward(self, z):
        """
        Forward process of VAE_Decoder
        Inputs:
            z : [tensor] latent variable of x;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        for layer_index in range(self.num_layers):
            z = F.relu(self.fc_decoder[layer_index](z))
        recon_x = torch.sigmoid(self.fc_decoder[-1](z))

        return recon_x
