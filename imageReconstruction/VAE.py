import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 p,l,
                 hidden_dims = None,
                 **kwargs):
        super(VAE, self).__init__()
        self.in_channels=in_channels
        self.latent_dim = latent_dim
        self.l=l

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding= 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*l*l, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*l*l, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * l*l)
        hidden_dims.reverse()

        for i in range(len(hidden_dims)-1 ):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,stride = 2,padding=1,output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,stride=2,padding=p[0],output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 3, padding= p[1]),
                            nn.Sigmoid())

    def encode(self, input):
        """
        input: (Tensor) Input tensor to encoder [B x C x H x W]
        return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        input z: (Tensor) [B x D]
        return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 32, self.l, self.l)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        input mu: (Tensor) Mean of the latent Gaussian [B x D]
        input logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), mu, log_var, z

    def visualize(self,input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        return result


