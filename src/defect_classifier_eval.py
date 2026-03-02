import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from defect_classifier_train import Classifier, ClsDataset

from torch.utils.data import DataLoader


DATA_PATH = "data/processed"


real = np.load(f"{DATA_PATH}/images.npy")
labels = np.load(f"{DATA_PATH}/labels.npy")


ds = ClsDataset(real, labels)

loader = DataLoader(ds, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Classifier().to(device)

model.load_state_dict(
    torch.load("checkpoints/classifier_real.pth", map_location=device)
)

model.eval()


preds = []
gt = []

with torch.no_grad():

    for x,l in loader:

        x = x.to(device)

        out = model(x)

        p = out.argmax(1).cpu().numpy()

        preds.extend(p)
        gt.extend(l.numpy())


cm = confusion_matrix(gt, preds)

plt.figure(figsize=(6,6))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.savefig("figures/confusion_matrix.png")

plt.show()


print(classification_report(gt, preds))