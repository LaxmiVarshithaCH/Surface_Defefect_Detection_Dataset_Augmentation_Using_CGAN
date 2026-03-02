import torch
import numpy as np
import matplotlib.pyplot as plt

from src.generator_cgan_surface import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


G = Generator().to(device)

G.load_state_dict(
    torch.load(
        "checkpoints/G_300.pth",
        map_location=device
    )
)

G.eval()


Z_DIM = 128


def generate(cls, n=6):

    imgs = []

    for i in range(n):

        z = torch.randn(1, Z_DIM).to(device)

        lab = torch.tensor([cls]).to(device)

        with torch.no_grad():

            img = G(z, lab)[0]

        img = img.cpu().permute(1,2,0).numpy()

        img = (img+1)/2

        imgs.append(img)

    return imgs



if __name__ == "__main__":

    imgs = generate(0,6)

    for i,img in enumerate(imgs):

        plt.subplot(1,6,i+1)

        plt.imshow(img)

        plt.axis("off")

    plt.show()