"""Write the synthetic head CT slices in demo_scans/.

The real RSNA scans used for training cannot be redistributed, so the public demo
ships hand-built phantoms instead: a skull, brain tissue with texture, ventricles,
and (in three of them) a patch of fresh blood at a realistic density (~65-75 HU).

They are NOT patient data and the model's scores on them are not a measure of
accuracy. The model never saw anything like them, so it is often unsure, which is
exactly when BleedAlert routes a case to human review.

    python scripts/make_phantoms.py
"""
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

OUT = Path(__file__).resolve().parents[1] / "demo_scans"
SIZE = 512


def blur(a, sigma):
    """Gaussian blur via FFT (no SciPy needed)."""
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


def phantom(kind, rng):
    yy, xx = np.mgrid[-1:1:SIZE * 1j, -1:1:SIZE * 1j]
    r = np.sqrt((xx / 0.72) ** 2 + (yy / 0.88) ** 2)
    hu = np.full((SIZE, SIZE), -1000.0)
    hu[r < 1.0] = 1500 + rng.normal(0, 60, (r < 1).sum())          # skull
    inner = r < 0.93
    tex = blur(rng.normal(0, 1, (SIZE, SIZE)), 6)
    tex /= tex.std()
    hu[inner] = 32 + 4 * tex[inner] + rng.normal(0, 3, inner.sum())  # brain
    hu[(r < 0.7) & inner] -= 6                                      # white matter
    vents = np.zeros_like(inner)
    for s in (-1, 1):
        vents |= (((xx - s * 0.1) / 0.07) ** 2 + ((yy + 0.05) / 0.25) ** 2) < 1
    hu[vents] = 6 + rng.normal(0, 2, vents.sum())                   # CSF
    sulci = (r > 0.86) & inner
    hu[sulci] = 15 + rng.normal(0, 3, sulci.sum())
    if kind == "subdural":      # crescent against the inner table
        cres = (r > 0.80) & (r < 0.93) & (np.abs(np.arctan2(yy, xx)) < 1.1)
        hu[cres] = 68 + rng.normal(0, 4, cres.sum())
    elif kind == "intraparenchymal":  # clot with a low-density edema rim
        d = np.sqrt((xx + 0.3) ** 2 + (yy - 0.1) ** 2)
        hu[(d < 0.2) & inner] = 22
        hu[d < 0.14] = 72 + rng.normal(0, 4, (d < 0.14).sum())
    elif kind == "intraventricular":  # blood layering in the ventricles
        layer = vents & (yy > 0.02)
        hu[layer] = 65 + rng.normal(0, 4, layer.sum())
    # Real scanners blur sharp boundaries slightly (partial volume); add it, then fresh noise.
    return blur(hu, 1.5) + rng.normal(0, 2, hu.shape)


def save(hu, path, label):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientID, ds.Modality = f"SYNTHETIC-{label.upper()}", "CT"
    ds.StudyDescription = "SYNTHETIC PHANTOM - NOT PATIENT DATA"
    ds.Rows = ds.Columns = SIZE
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
    ds.ImagePositionPatient = [0, 0, 50.0]
    ds.PixelData = np.clip(hu + 1024, -32768, 32767).astype(np.int16).tobytes()
    ds.save_as(path, enforce_file_format=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for kind in ("normal", "subdural", "intraparenchymal", "intraventricular"):
        save(phantom(kind, rng), OUT / f"synthetic_{kind}.dcm", kind)
    print(f"wrote 4 synthetic slices to {OUT}")
