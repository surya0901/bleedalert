"""Model loading, prediction and Grad-CAM heatmaps.

If neither bleedalert.pt nor bleedalert_weights.npz exists (before you finish training), the app runs in
DEMO MODE with a fake scorer so the frontend and backend team can build in parallel.
"""
import base64
import io
import os
import random

import numpy as np
from PIL import Image

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
HERE = os.path.dirname(__file__)
WEIGHTS_NPZ = os.path.join(HERE, "bleedalert_weights.npz")  # fp16 weights, small enough for GitHub
MODEL_PATH = os.path.join(HERE, "bleedalert.pt")            # full checkpoint straight from Kaggle


def build_model():
    import torch.nn as nn
    import torchvision

    m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(LABELS))
    return m


class Predictor:
    def __init__(self):
        self.model = None
        self.source = None
        if os.path.exists(MODEL_PATH):
            self.source = MODEL_PATH
        elif os.path.exists(WEIGHTS_NPZ):
            self.source = WEIGHTS_NPZ
        self.demo = self.source is None
        if not self.demo:
            import torch

            if self.source == MODEL_PATH:
                state = torch.load(MODEL_PATH, map_location="cpu")
            else:
                arrays = np.load(WEIGHTS_NPZ)
                state = {k: torch.from_numpy(arrays[k].astype(np.float32) if arrays[k].dtype == np.float16
                                             else arrays[k]) for k in arrays.files}
            self.model = build_model()
            self.model.load_state_dict(state)
            self.model.eval()

    def predict(self, img):
        """Returns ({label: probability}, heatmap_png_base64)."""
        if self.demo:
            return self._demo_predict(img)

        import torch
        from ct_utils import to_tensor

        x = to_tensor(img)
        target_layer = self.model.features[-1]
        acts, grads = {}, {}

        def keep(module, inputs, output):
            acts["a"] = output
            output.register_hook(lambda g: grads.setdefault("g", g))

        handle = target_layer.register_forward_hook(keep)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].detach()
            self.model.zero_grad()
            logits[0, 0].backward()  # explain the "any bleed" output
        handle.remove()

        weights = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts["a"]).sum(dim=1))[0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        scores = {lab: float(p) for lab, p in zip(LABELS, probs)}
        return scores, overlay(img, cam)

    def _demo_predict(self, img):
        seed = int(np.asarray(img).sum()) % 10_000
        rng = random.Random(seed)
        any_p = rng.random()
        scores = {"any": any_p}
        for lab in LABELS[1:]:
            scores[lab] = any_p * rng.random()
        cam = np.zeros((8, 8))
        cam[rng.randint(1, 6), rng.randint(1, 6)] = 1.0
        return scores, overlay(img, cam)


def overlay(img, cam):
    """Blend a red heatmap over the scan and return a base64 PNG."""
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32) / 255.0
    base = np.asarray(img, dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    alpha = (heat * 0.55)[..., None]
    out = (base * (1 - alpha) + red * alpha).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
