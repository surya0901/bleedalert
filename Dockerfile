# Runs BleedAlert on any container host. Used for the free Hugging Face Space demo.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BLEEDALERT_DB=/tmp/bleedalert.db
WORKDIR /app

# CPU-only PyTorch keeps the image small and runs fine on free hardware.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.py ct_utils.py simulate.py index.html bleedalert_weights.npz ./
COPY demo_scans ./demo_scans

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
