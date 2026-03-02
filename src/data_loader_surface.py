import numpy as np
import torch
from torch.utils.data import Dataset


class SurfaceDataset(Dataset):

    def __init__(self, data_dir):

        self.X = np.load(f"{data_dir}/images.npy")
        self.y = np.load(f"{data_dir}/labels.npy")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        img = self.X[i]
        lab = self.y[i]

        img = torch.tensor(img).permute(2,0,1)

        return img, lab