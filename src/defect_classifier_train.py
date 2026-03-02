import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader


DATA_PATH = "data/processed"


class ClsDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        img = torch.tensor(self.X[i]).permute(2,0,1).float()
        lab = self.y[i]

        return img, lab



class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(128*16*16,256),
            nn.ReLU(),

            nn.Linear(256,6)
        )

    def forward(self,x):
        return self.net(x)



def train_classifier(X, y):

    ds = ClsDataset(X,y)

    dl = DataLoader(ds, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier().to(device)

    opt = torch.optim.Adam(model.parameters(), 1e-3)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):

        for x,l in dl:

            x = x.to(device)
            l = l.to(device)

            opt.zero_grad()

            out = model(x)

            loss = loss_fn(out,l)

            loss.backward()

            opt.step()

    return model



if __name__ == "__main__":

    real = np.load(f"{DATA_PATH}/images.npy")
    labels = np.load(f"{DATA_PATH}/labels.npy")

    model = train_classifier(real, labels)

    torch.save(model.state_dict(), "checkpoints/classifier_real.pth")