import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import albumentations as A


DATA_ROOT = "data/raw/NEU-DET/validation/images"

OUT_DIR = "data/processed"

IMG_SIZE = 128

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

os.makedirs(OUT_DIR, exist_ok=True)


augment = A.Compose([
    A.Resize(128,128),
    A.HorizontalFlip(p=0.5),
])


images = []
labels = []

AUG_PER_IMAGE = 4


for class_id, cls in enumerate(CLASSES):

    folder = os.path.join(DATA_ROOT, cls)

    files = os.listdir(folder)

    for f in tqdm(files):

        path = os.path.join(folder, f)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(AUG_PER_IMAGE):

            aug = augment(image=img)["image"]

            aug = aug.astype(np.float32) / 127.5 - 1.0

            images.append(aug)
            labels.append(class_id)


images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)

np.save(f"{OUT_DIR}/images.npy", images)
np.save(f"{OUT_DIR}/labels.npy", labels)

class_map = {i:c for i,c in enumerate(CLASSES)}

with open(f"{OUT_DIR}/class_map.json","w") as f:
    json.dump(class_map, f)

print("Saved")