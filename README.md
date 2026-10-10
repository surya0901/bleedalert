# BleedAlert

AI triage for head CT scans. Flags a likely brain bleed, moves it to the top of the
neurologist's worklist, shows a heatmap, writes a short summary, and speaks an alert.
Built for the 2026 Rutgers Health Hack (Sevaro track). Triage support only, not a diagnosis.

**Total cost: $0.** Everything below uses free tiers.

| Piece | Free option |
|---|---|
| Training GPU | Kaggle Notebooks (about 30 free GPU hours a week) |
| Data | RSNA Intracranial Hemorrhage dataset on Kaggle |
| Model | PyTorch EfficientNet (pretrained weights are free) |
| Backend | FastAPI + SQLite, runs on your laptop |
| Summary | Gemini API free tier (optional, works without it) |
| Voice alert | Browser speech, built in |
| Hosting for judges | Run on a laptop at the table, or free tiers like Render |
| Sponsor cloud bonus | Azure for Students ($100 credit, no card) or AWS credits from the event |

## 1. Run it locally (5 minutes)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000 and drop in head CT slices (.dcm). The trained model weights
(`bleedalert_weights.npz`, half precision) are included in the repo, so the header shows "Model loaded." If the file is missing, the app
falls back to **DEMO MODE** with fake scores.

## 2. Train the real model on Kaggle (free GPU, about 1.5 hours)

1. Make a Kaggle account and verify your phone (needed for GPU).
2. Open the RSNA Intracranial Hemorrhage Detection competition page and accept the rules.
3. Create a new notebook, add that competition as input, set Accelerator to GPU and Internet On.
4. Paste `train_kaggle.py` into a cell and run it.
5. From the Output tab, download `bleedalert.pt` into the project folder and keep the `demo_scans/`
   files handy for the live demo.
6. Restart the server. The header switches from DEMO MODE to Model loaded.

The script prints its test metrics and saves them to `metrics.json`.

## Model results (patient-level test set)

Tested on 17,978 slices from 2,780 patients the model never saw during training
(every patient is in training or testing, never both).

| Metric | Result |
|---|---|
| AUC, any bleed | 0.978 |
| Real bleeds flagged URGENT or NEEDS HUMAN REVIEW | 95% |
| Real bleeds flagged URGENT | 90% |
| Normal slices correctly left ROUTINE | 88% |
| Slices sent to human review (model unsure) | 6% |

Subtype AUC: intraventricular 0.984, intraparenchymal 0.971, subdural 0.947,
subarachnoid 0.931, epidural 0.919. Scores are per slice on a balanced 50/50 set, so they are
not a clinical validation. Triage support only.

## Demo scans

`demo_scans/` holds 16 de-identified RSNA slices from TEST patients (8 bleed, 8 normal), so the
live demo never shows an image the model trained on. Drag them into the dashboard.

## 3. Optional: Gemini summaries

Get a free key at aistudio.google.com, then before starting the server:

```bash
export GEMINI_API_KEY=your_key_here     # Windows: set GEMINI_API_KEY=your_key_here
```

## How triage works

| Bleed probability | Status | What happens |
|---|---|---|
| 65% or higher | URGENT | Top of the queue, voice alert |
| 35% to 65% | NEEDS HUMAN REVIEW | Model is unsure, says so |
| below 35% | ROUTINE | Normal order |

Thresholds live at the top of `app.py`.

## Project layout

```
app.py            API: upload, worklist, review, serves the dashboard
model.py          model loading, prediction, Grad-CAM heatmap, demo mode
ct_utils.py       DICOM to 3 window image
index.html        worklist dashboard
train_kaggle.py   Kaggle training script
bleedalert_weights.npz  trained weights, fp16 (a full bleedalert.pt from Kaggle also works)
```

## Team split

- ML lead: run the Kaggle training now, it takes the longest.
- Backend lead: Gemini summary, thresholds, maybe a day of scans simulation endpoint.
- Frontend lead: polish `index.html`.
- Pitch lead: talk to clinicians and Sevaro mentors, build slides, record a backup video.

## Notes

- Only public research data. No real patient data.
- `bleedalert_weights.npz` (about 8 MB) is committed so the app runs out of the box. Never commit the dataset.
