import torch
import torch.nn as nn

IMG_SIZE = 128
Z_DIM = 128
NUM_CLASSES = 6
EMB_DIM = 32
CHANNELS = 3


class CBN(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.embed = nn.Embedding(
            NUM_CLASSES,
            num_features * 2
        )

        self.embed.weight.data[:, :num_features].fill_(1)
        self.embed.weight.data[:, num_features:].zero_()


    def forward(self, x, y):

        out = self.bn(x)

        gamma, beta = self.embed(y).chunk(2, 1)

        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        out = gamma * out + beta

        return out



class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(NUM_CLASSES, EMB_DIM)

        self.fc = nn.Linear(Z_DIM + EMB_DIM, 512 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, CHANNELS, 4, 2, 1)

        self.cbn1 = CBN(256)
        self.cbn2 = CBN(128)
        self.cbn3 = CBN(64)
        self.cbn4 = CBN(32)

        self.relu = nn.ReLU(True)


    def forward(self, z, labels):

        y = self.embed(labels)

        x = torch.cat([z, y], dim=1)

        x = self.fc(x)

        x = x.view(-1, 512, 4, 4)

        x = self.deconv1(x)
        x = self.cbn1(x, labels)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.cbn2(x, labels)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.cbn3(x, labels)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.cbn4(x, labels)
        x = self.relu(x)

        x = self.deconv5(x)

        img = torch.tanh(x)

        return img