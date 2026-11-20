# Clinical Intelligence Platform — Backend

## Install

Run from `D:\Clinical_Project` with your existing virtual environment activated:

```powershell
pip install -r backend\requirements.txt
```

## Start

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API testing.

## CT ZIP input

Create a ZIP containing sequential PNG CT slices. PNGs may be inside subfolders; the backend extracts them and sorts the resulting filenames. Use consistent slice naming such as `slice-0001.png`, `slice-0002.png`, etc.

The current model was trained on nodule-centered 8-slice patches at 128×128 resolution. The backend resizes input slices to 128×128, creates overlapping patches with stride 4, averages overlapping probability predictions, and writes probability maps, binary masks and overlays.

**Important limitations**
- This is an experimental research workflow, not a clinically validated full-volume segmentation system.
- The training patches were nodule-centered. An arbitrary full-lung scan may have a different input distribution.
- PNG files do not preserve DICOM pixel spacing or reliable Hounsfield units. Do not infer physical nodule size, volume or clinical stage from these outputs.
- Cancer risk output is a tabular model estimate, not a diagnosis. A low score must not be used to rule out CT analysis.
- Do not expose this development API publicly or upload identifiable patient data without appropriate security, consent and governance.

## API

- `GET /health`
- `POST /predict/cancer-risk`
- `POST /predict/ct-segmentation` — multipart ZIP field `file`, optional form field `threshold` (default 0.20)
- `GET /results/{job_id}` — download generated result ZIP
