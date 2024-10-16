import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # input batch_size, 110, 1, 1
        self.model = nn.Sequential(
            nn.ConvTranspose2d(110, 256, 4, 1, 0, bias=False), # batch_size, 256, 4, 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # batch_size, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # batch_size, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), # batch_size, 1, 32, 32
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z (batch_size, 100) gen_labels (batch_size, 10)
        z = torch.cat([z, labels], 1).view(-1, 110, 1, 1)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(11, 64, 4, 2, 1, bias=False), # batch_size, 64, 16, 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # batch_size, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # batch_size, 256, 4, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False), # batch_size, 1, 1, 1
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        disc_labels = labels.view(x.size(0), 10, 1, 1).expand(-1, -1, 32, 32)
        # x (batch_size, 1, 28, 28) labels (batch_size, 10, 32, 32)
        x = torch.cat([x, disc_labels], 1) # batch_size, 11, 32, 32
        output = self.model(x)
        return output.view(-1, 1)