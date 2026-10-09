"""Turn a head CT slice (DICOM or image) into a 3 channel tensor the model expects.

Radiologists look at the same CT with different "windows" (contrast settings).
We stack three of them as RGB channels, a common trick from the RSNA competition:
  channel 0: brain window      (center 40,  width 80)
  channel 1: subdural window   (center 80,  width 200)
  channel 2: bone window       (center 600, width 2800)
"""
import io

import numpy as np
from PIL import Image

IMG_SIZE = 256
WINDOWS = [(40, 80), (80, 200), (600, 2800)]


def _window(hu, center, width):
    lo, hi = center - width / 2, center + width / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def dicom_to_rgb(raw_bytes):
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = pixels * slope + intercept
    channels = [_window(hu, c, w) for c, w in WINDOWS]
    rgb = (np.stack(channels, axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def study_key(raw_bytes, filename):
    """Which study (patient scan) a slice belongs to.

    DICOM slices carry a StudyInstanceUID, so 40 uploaded slices of one head CT
    land in one worklist row. Plain images use their file name as the study.
    """
    if filename.lower().endswith(".dcm"):
        try:
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(raw_bytes), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            if uid:
                return uid
        except Exception:
            pass
    return filename


def load_scan(raw_bytes, filename):
    """Accept a .dcm file or an already exported PNG/JPG."""
    if filename.lower().endswith(".dcm"):
        img = dicom_to_rgb(raw_bytes)
    else:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return img.resize((IMG_SIZE, IMG_SIZE))


def to_tensor(img):
    import torch

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
