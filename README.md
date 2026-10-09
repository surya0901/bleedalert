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

Open http://localhost:8000 and drop in any scan. Without a trained model it runs in
**DEMO MODE** with fake scores, so frontend and backend work can start right away.

## 2. Train the real model on Kaggle (free GPU, about 1.5 hours)

1. Make a Kaggle account and verify your phone (needed for GPU).
2. Open the RSNA Intracranial Hemorrhage Detection competition page and accept the rules.
3. Create a new notebook, add that competition as input, set Accelerator to GPU and Internet On.
4. Paste `train_kaggle.py` into a cell and run it.
5. From the Output tab, download `bleedalert.pt` into the project folder and keep the `demo_scans/`
   files handy for the live demo.
6. Restart the server. The header switches from DEMO MODE to Model loaded.

The script prints a validation AUC. Put that number on a slide.

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
bleedalert.pt     trained model (download from Kaggle, not committed)
```

## Team split

- ML lead: run the Kaggle training now, it takes the longest.
- Backend lead: Gemini summary, thresholds, maybe a day of scans simulation endpoint.
- Frontend lead: polish `index.html`.
- Pitch lead: talk to clinicians and Sevaro mentors, build slides, record a backup video.

## Notes

- Only public research data. No real patient data.
- Do not commit `bleedalert.pt` or the dataset to GitHub (too big). Share the model by Google Drive.
