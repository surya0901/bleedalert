"""Export the trained model to ONNX for in-browser inference (web/ demo).

The exported graph returns:
  probs  (1, 6)    sigmoid probabilities, same order as model.LABELS
  cam    (1, 8, 8) heatmap for "any bleed", normalized to 0..1

Why the CAM can be computed without gradients: the Grad-CAM target layer
(features[-1]) feeds global average pooling and then a single linear layer, so the
gradient of the "any" logit with respect to every feature-map position is that
class's weight divided by H*W. Grad-CAM therefore equals ReLU(sum_k w_k * A_k)
up to a constant, which disappears when normalizing. Same heatmap, no backward pass.

    python scripts/export_onnx.py           # writes web/model.onnx
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import Predictor  # noqa: E402


class WebModel(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.features, self.avgpool = net.features, net.avgpool
        self.fc = net.classifier[1]

    def forward(self, x):
        a = self.features(x)                                   # (1, 1280, 8, 8)
        logits = self.fc(torch.flatten(self.avgpool(a), 1))
        cam = torch.relu(torch.einsum("k,bkhw->bhw", self.fc.weight[0], a))
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return torch.sigmoid(logits), cam


if __name__ == "__main__":
    p = Predictor()
    assert not p.demo, "trained weights not found"
    m = WebModel(p.model).eval()
    out = os.path.join(ROOT, "web", "model.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.onnx.export(m, torch.zeros(1, 3, 256, 256), out, input_names=["image"],
                      output_names=["probs", "cam"], opset_version=17, dynamo=False)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
