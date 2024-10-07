import torch
from torch import nn
from sklearn import svm



class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encode_layers = nn.Sequential(
            nn.Linear(input_dim, latent_dim[0]),
            nn.ReLU(),
            nn.Linear(latent_dim[0], latent_dim[1]),
            nn.ReLU(),
            nn.Linear(latent_dim[1], latent_dim[2]),
            nn.ReLU(),
        )
        self.mean = nn.Linear(latent_dim[2], latent_dim[3])
        self.log_var = nn.Linear(latent_dim[2], latent_dim[3])

        self.decode_layers = nn.Sequential(
            nn.Linear(latent_dim[3], latent_dim[2]),
            nn.ReLU(),
            nn.Linear(latent_dim[2], latent_dim[1]),
            nn.ReLU(),
            nn.Linear(latent_dim[1], latent_dim[0]),
            nn.ReLU(),
            nn.Linear(latent_dim[0], input_dim),
            nn.Sigmoid()
        )
 
    def encode(self, x):
        fore1 = self.encode_layers(x)
        mean = self.mean(fore1)
        log_var = self.log_var(fore1)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        recon_x = self.decode_layers(z)
        return recon_x

    
    def forward(self, x):
        org_size = x.size()
        batch_size = org_size[0]
        x = x.view((batch_size, -1))
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decode(z).view(org_size)
        return z, recon_x, mean, log_var
    

if __name__ == "__main__":
    input_dim = 784
    latent_dim = [512, 256, 128, 64]
    output_dim = 10
    classifier_dim = [64, 32]
    model = VAE(input_dim, latent_dim)
    input = torch.rand((1, 1, 784))
    output = model(input)
    print(output)