from src.monitor.log_usage import log_usage
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from generator_cgan_surface import Generator
from config import load_config


cfg = load_config()

Z_DIM = cfg["z_dim"]
CLASSES = cfg["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)

G.load_state_dict(
    torch.load(cfg["checkpoint"], map_location=device)
)

G.eval()


st.title("Surface Defect Generator")


cls = st.selectbox(
    "Defect",
    range(len(CLASSES)),
    format_func=lambda x: CLASSES[x]
)

n = st.slider("Count", 1, 100, 6)


def gen():

    imgs = []

    for i in range(n):

        z = torch.randn(1, Z_DIM).to(device)

        lab = torch.tensor([cls]).to(device)

        with torch.no_grad():

            img = G(z, lab)[0]

        img = img.cpu().permute(1,2,0).numpy()

        img = (img + 1)/2

        imgs.append(img)

    return imgs


if st.button("Generate"):

    log_usage(
        source="streamlit",
        cls=CLASSES[cls],
        n=n,
    )

    imgs = gen()

    cols = st.columns(n)

    for i in range(n):

        cols[i].image(imgs[i])