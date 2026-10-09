"""Fine tune BleedAlert on Kaggle's FREE GPU.

How to run (free):
 1. kaggle.com -> Create -> New Notebook
 2. Right panel: Add Input -> search "RSNA Intracranial Hemorrhage Detection" -> add the competition data
    (you must click "Join Competition" / accept rules once on the competition page)
 3. Settings: Accelerator = GPU T4 x2 (or P100), Internet = On
 4. Paste this whole file into one cell and run
 5. When done, download /kaggle/working/bleedalert.pt and put it in the project folder next to app.py

Uses a 20k image subset and 3 epochs so it finishes in about 1 to 1.5 hours.
Raise N_TRAIN if you have time.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

DATA = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
IMG_DIR = f"{DATA}/stage_2_train"
LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
N_TRAIN, N_VAL, EPOCHS, BATCH, IMG_SIZE = 20_000, 3_000, 3, 64, 256
WINDOWS = [(40, 80), (80, 200), (600, 2800)]
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)

# Labels file has rows like "ID_abc123_epidural, 0". Pivot to one row per image.
df = pd.read_csv(f"{DATA}/stage_2_train.csv")
df[["img", "label"]] = df["ID"].str.rsplit("_", n=1, expand=True)
df = df.drop_duplicates(["img", "label"]).pivot(index="img", columns="label", values="Label")[LABELS]

# Balance: half with a bleed, half without, so the model sees enough positives
pos = df[df["any"] == 1].sample(n=(N_TRAIN + N_VAL) // 2, random_state=0)
neg = df[df["any"] == 0].sample(n=(N_TRAIN + N_VAL) // 2, random_state=0)
data = pd.concat([pos, neg]).sample(frac=1, random_state=0)
train_df, val_df = data.iloc[:N_TRAIN], data.iloc[N_TRAIN:]


def window(hu, c, w):
    return np.clip((hu - (c - w / 2)) / w, 0, 1)


class CTData(Dataset):
    def __init__(self, frame, train):
        self.ids, self.y, self.train = frame.index.tolist(), frame.values.astype(np.float32), train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        try:
            ds = pydicom.dcmread(f"{IMG_DIR}/{self.ids[i]}.dcm")
            hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            rgb = np.stack([window(hu, c, w) for c, w in WINDOWS], -1)
        except Exception:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)  # a few files are corrupt
        img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, np.float32) / 255.0
        if self.train and random.random() < 0.5:
            arr = arr[:, ::-1].copy()  # left right flip
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32), torch.tensor(self.y[i])


train_dl = DataLoader(CTData(train_df, True), batch_size=BATCH, shuffle=True, num_workers=4)
val_dl = DataLoader(CTData(val_df, False), batch_size=BATCH, num_workers=4)

# Start from ImageNet weights (pretrained), swap the last layer for our 6 labels
model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(train_dl))
loss_fn = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

best = 0
for epoch in range(EPOCHS):
    model.train()
    for step, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0:
            print(f"epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}", flush=True)

    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in val_dl:
            preds.append(torch.sigmoid(model(x.to(device))).cpu())
            ys.append(y)
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    auc = roc_auc_score(ys[:, 0], preds[:, 0])
    print(f"epoch {epoch} validation AUC (any bleed): {auc:.3f}")
    if auc > best:
        best = auc
        torch.save(model.state_dict(), "/kaggle/working/bleedalert.pt")
        print("saved bleedalert.pt")

# Save some sample validation scans for the live demo (mix of bleed and no bleed)
os.makedirs("/kaggle/working/demo_scans", exist_ok=True)
for img_id in list(val_df[val_df["any"] == 1].index[:6]) + list(val_df[val_df["any"] == 0].index[:6]):
    tag = "bleed" if val_df.loc[img_id, "any"] == 1 else "normal"
    os.system(f"cp {IMG_DIR}/{img_id}.dcm /kaggle/working/demo_scans/{tag}_{img_id}.dcm")
print(f"Done. Best AUC {best:.3f}. Download bleedalert.pt and demo_scans/ from the Output tab.")
