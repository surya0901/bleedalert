"""Train BleedAlert on Kaggle's FREE GPU, with an honest patient-level test set.

How to run (free):
 1. On the RSNA Intracranial Hemorrhage Detection competition page: Code -> New Notebook
    (accept the competition rules once; Kaggle needs a verified phone for GPU)
 2. Session options: Accelerator = GPU T4 x2, Internet = On
 3. Paste this file into one cell, then Save Version -> Save & Run All
 4. From the Output tab download bleedalert.pt, metrics.json and demo_scans/
    and put bleedalert.pt next to app.py

Every patient is EITHER in training OR in testing, never both, so the test score
reflects how the model does on people it has never seen.
"""
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Find the data wherever Kaggle mounted it (the folder layout has changed over the years)
_csv = [p for depth in range(1, 5) for p in glob.glob("/kaggle/input/" + "*/" * depth + "stage_2_train.csv")]
if not _csv:
    sys.exit("stage_2_train.csv not found. Is the RSNA competition data added as an input?")
DATA = os.path.dirname(_csv[0])
IMG_DIR = f"{DATA}/stage_2_train"
print("Using data from", DATA, flush=True)

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
N_POS, N_NEG = 60_000, 60_000   # slices sampled before the patient split
TEST_FRACTION = 0.15            # share of PATIENTS held out for testing
EPOCHS, BATCH, IMG_SIZE = 3, 64, 256
WINDOWS = [(40, 80), (80, 200), (600, 2800)]
URGENT, UNSURE_LOW = 0.65, 0.35  # same thresholds as app.py
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)
np.random.seed(0)

df = pd.read_csv(f"{DATA}/stage_2_train.csv")
df[["img", "label"]] = df["ID"].str.rsplit("_", n=1, expand=True)
df = df.drop_duplicates(["img", "label"]).pivot(index="img", columns="label", values="Label")[LABELS]
data = pd.concat([
    df[df["any"] == 1].sample(n=N_POS, random_state=0),
    df[df["any"] == 0].sample(n=N_NEG, random_state=0),
])


def patient_of(img_id):
    try:
        ds = pydicom.dcmread(f"{IMG_DIR}/{img_id}.dcm", stop_before_pixels=True,
                             specific_tags=["PatientID"])
        return str(ds.PatientID)
    except Exception:
        return None


print(f"Reading patient IDs for {len(data)} slices...", flush=True)
with Pool(8) as pool:
    data["patient"] = pool.map(patient_of, data.index.tolist(), chunksize=500)
data = data.dropna(subset=["patient"])

patients = data["patient"].unique().tolist()
random.shuffle(patients)
test_patients = set(patients[: int(len(patients) * TEST_FRACTION)])
is_test = data["patient"].isin(test_patients)
train_df, val_df = data[~is_test].sample(frac=1, random_state=0), data[is_test]
print(f"{len(patients)} patients: {len(patients) - len(test_patients)} train, {len(test_patients)} test. "
      f"{len(train_df)} train slices, {len(val_df)} test slices", flush=True)
assert not set(train_df["patient"]) & set(val_df["patient"]), "patient leaked into both splits"


def window(hu, c, w):
    return np.clip((hu - (c - w / 2)) / w, 0, 1)


class CTData(Dataset):
    def __init__(self, frame, train):
        self.ids = frame.index.tolist()
        self.y = frame[LABELS].values.astype(np.float32)
        self.train = train

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

model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(train_dl))
loss_fn = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda")

best, best_metrics = 0, {}
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
        if step % 200 == 0:
            print(f"epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}", flush=True)

    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in val_dl:
            preds.append(torch.sigmoid(model(x.to(device))).cpu())
            ys.append(y)
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    p, t = preds[:, 0], ys[:, 0].astype(bool)
    metrics = {
        "epoch": epoch,
        "test_patients": len(test_patients),
        "test_slices": int(len(t)),
        "auc_any_bleed": round(float(roc_auc_score(t, p)), 4),
        "auc_per_subtype": {lab: round(float(roc_auc_score(ys[:, k], preds[:, k])), 4)
                            for k, lab in enumerate(LABELS) if 0 < ys[:, k].sum() < len(ys)},
        # Of real bleeds, how many the app marks URGENT / at least NEEDS HUMAN REVIEW
        "sensitivity_urgent": round(float((p[t] >= URGENT).mean()), 4),
        "sensitivity_urgent_or_review": round(float((p[t] >= UNSURE_LOW).mean()), 4),
        # Of normal slices, how many the app correctly leaves ROUTINE
        "specificity_routine": round(float((p[~t] < UNSURE_LOW).mean()), 4),
        "share_sent_to_human_review": round(float(((p >= UNSURE_LOW) & (p < URGENT)).mean()), 4),
    }
    print(json.dumps(metrics), flush=True)
    if metrics["auc_any_bleed"] > best:
        best, best_metrics = metrics["auc_any_bleed"], metrics
        torch.save(model.state_dict(), "/kaggle/working/bleedalert.pt")
        print("saved bleedalert.pt", flush=True)

with open("/kaggle/working/metrics.json", "w") as f:
    json.dump(best_metrics, f, indent=2)

# Demo scans from TEST patients only, so the live demo never shows a training image
os.makedirs("/kaggle/working/demo_scans", exist_ok=True)
for tag, rows in (("bleed", val_df[val_df["any"] == 1]), ("normal", val_df[val_df["any"] == 0])):
    for img_id in rows.index[:8]:
        os.system(f"cp {IMG_DIR}/{img_id}.dcm /kaggle/working/demo_scans/{tag}_{img_id}.dcm")
print(f"Done. Best patient-level test AUC {best:.3f}. See metrics.json.", flush=True)
