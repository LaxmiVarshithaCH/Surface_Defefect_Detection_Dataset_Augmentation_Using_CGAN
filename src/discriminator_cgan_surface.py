import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

NUM_CLASSES = 6


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
        )

        self.flatten = nn.Flatten()

        self._feat_dim = 512 * 4 * 4

        self.adv = spectral_norm(nn.Linear(self._feat_dim, 1))
        self.cls = spectral_norm(nn.Linear(self._feat_dim, NUM_CLASSES))


    def forward(self, x):

        f = self.conv(x)

        f = self.flatten(f)

        adv = self.adv(f)
        cls = self.cls(f)

        return adv, cls