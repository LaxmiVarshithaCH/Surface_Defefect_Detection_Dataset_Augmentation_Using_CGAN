import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

from generator_cgan_surface import Generator
from discriminator_cgan_surface import Discriminator
from data_loader_surface import SurfaceDataset

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast


DATA_PATH = "data/processed"

SAVE_DIR = "checkpoints"

SAMPLE_DIR = "samples"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


images = np.load(f"{DATA_PATH}/images.npy")
labels = np.load(f"{DATA_PATH}/labels.npy")


dataset = SurfaceDataset(DATA_PATH)


class_counts = np.bincount(labels)

weights = 1. / class_counts

sample_weights = weights[labels]


sampler = WeightedRandomSampler(
    sample_weights,
    len(sample_weights),
    replacement=True
)


loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,
    num_workers=0
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


G = Generator().to(device)
D = Discriminator().to(device)


lr = 1e-4

opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0,0.9))
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0,0.9))


adv_loss = nn.BCEWithLogitsLoss()
cls_loss = nn.CrossEntropyLoss()


scaler_G = GradScaler("cuda")
scaler_D = GradScaler("cuda")


Z_DIM = 128
NUM_CLASSES = 6

EPOCHS = 300

real_label = 0.9


def diff_augment(x):

    if torch.rand(1) < 0.5:
        x = torch.flip(x, [3])

    noise = torch.randn_like(x) * 0.03
    x = x + noise

    x = torch.clamp(x, -1, 1)

    return x


def save_ckpt(epoch):

    torch.save(G.state_dict(), f"{SAVE_DIR}/G_{epoch}.pth")
    torch.save(D.state_dict(), f"{SAVE_DIR}/D_{epoch}.pth")


for epoch in range(EPOCHS):

    pbar = tqdm(loader)

    for real_img, real_lab in pbar:

        real_img = real_img.to(device).float()
        real_img = diff_augment(real_img)

        real_lab = real_lab.to(device)

        b = real_img.size(0)

        valid = torch.full((b,1), real_label).to(device)
        fake = torch.zeros((b,1)).to(device)

        # D

        opt_D.zero_grad()

        z = torch.randn(b, Z_DIM).to(device)
        fake_lab = torch.randint(0, NUM_CLASSES, (b,)).to(device)

        with autocast("cuda"):

            fake_img = G(z, fake_lab)
            fake_img = diff_augment(fake_img)

            adv_real, cls_real = D(real_img)
            adv_fake, cls_fake = D(fake_img.detach())

            loss_real = adv_loss(adv_real, valid)
            loss_fake = adv_loss(adv_fake, fake)

            loss_cls = cls_loss(cls_real, real_lab)

            loss_D = loss_real + loss_fake + loss_cls

        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()


        # G

        opt_G.zero_grad()

        z = torch.randn(b, Z_DIM).to(device)
        gen_lab = torch.randint(0, NUM_CLASSES, (b,)).to(device)

        with autocast("cuda"):

            gen_img = G(z, gen_lab)

            adv, cls = D(gen_img)

            loss_G_adv = adv_loss(adv, valid)
            loss_G_cls = cls_loss(cls, gen_lab)

            loss_G = loss_G_adv + loss_G_cls

        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()


        pbar.set_description(
            f"E{epoch} D:{loss_D.item():.3f} G:{loss_G.item():.3f}"
        )


    if epoch % 50 == 0:
        save_ckpt(epoch)