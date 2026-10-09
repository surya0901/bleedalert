"""Model loading, prediction and Grad-CAM heatmaps.

If bleedalert.pt is missing (before you finish training), the app runs in
DEMO MODE with a fake scorer so the frontend and backend team can build in parallel.
"""
import base64
import io
import os
import random

import numpy as np
from PIL import Image

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "bleedalert.pt")


def build_model():
    import torch.nn as nn
    import torchvision

    m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(LABELS))
    return m


class Predictor:
    def __init__(self):
        self.demo = not os.path.exists(MODEL_PATH)
        self.model = None
        if not self.demo:
            import torch

            self.model = build_model()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            self.model.eval()

    def predict(self, img):
        """Returns ({label: probability}, heatmap_png_base64)."""
        if self.demo:
            return self._demo_predict(img)

        import torch
        from ct_utils import to_tensor

        x = to_tensor(img).requires_grad_(False)
        target_layer = self.model.features[-1]
        acts, grads = {}, {}
        h1 = target_layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
        h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault("g", go[0]))

        logits = self.model(x)
        probs = torch.sigmoid(logits)[0]
        self.model.zero_grad()
        logits[0, 0].backward()  # explain the "any bleed" output
        h1.remove()
        h2.remove()

        weights = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts["a"]).sum(dim=1))[0].detach().numpy()
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
