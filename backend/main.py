from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.config import OUTPUT_DIR
from backend.services.cancer_classifier import CancerClassifier
from backend.services.ct_segmentation import CTSegmentationService

classifier = None
segmenter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, segmenter
    classifier = CancerClassifier()
    segmenter = CTSegmentationService()
    classifier.load()
    segmenter.load()
    yield


app = FastAPI(
    title="Clinical Intelligence Platform API",
    description="Research prototype for tabular cancer-risk estimation and CT nodule segmentation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Restrict this list to your actual frontend origin before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientFeatures(BaseModel):
    Age: float = Field(gt=0, le=120)
    Gender: float
    BMI: float = Field(gt=0, le=100)
    Smoking: float
    GeneticRisk: float
    PhysicalActivity: float = Field(ge=0)
    AlcoholIntake: float = Field(ge=0)
    CancerHistory: float


@app.get("/")
def root():
    return {
        "name": "Clinical Intelligence Platform API",
        "status": "running",
        "disclaimer": "Research decision-support prototype; not a medical diagnosis.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cancer_classifier_loaded": classifier is not None and classifier.is_loaded,
        "ct_segmenter_loaded": segmenter is not None and segmenter.is_loaded,
        "device": segmenter.device_name if segmenter else None,
    }


@app.post("/predict/cancer-risk")
def predict_cancer_risk(patient: PatientFeatures):
    try:
        return classifier.predict(patient.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/ct-segmentation")
async def predict_ct_segmentation(
    file: UploadFile = File(...),
    threshold: float = Form(0.20),
):
    if not 0.0 < threshold < 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a ZIP containing sequential PNG CT slices.")

    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    input_dir = job_dir / "input"
    result_dir = job_dir / "result"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    archive_path = job_dir / "upload.zip"
    try:
        with archive_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        # Reject unsafe paths and non-PNG files; extract only PNG slices.
        with zipfile.ZipFile(archive_path) as archive:
            png_entries = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise HTTPException(status_code=400, detail="ZIP contains an unsafe path.")
                if member.suffix.lower() == ".png":
                    png_entries.append(info)

            if not png_entries:
                raise HTTPException(status_code=400, detail="ZIP contains no PNG images.")

            # Flatten to unique generated names; the service sorts by original basename.
            for index, info in enumerate(png_entries):
                safe_name = f"{index:06d}_{Path(info.filename).name}"
                with archive.open(info) as src, (input_dir / safe_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        result = segmenter.predict_folder(
            input_dir=input_dir,
            output_dir=result_dir,
            threshold=threshold,
        )

        result_zip = job_dir / "segmentation_results.zip"
        with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in result_dir.iterdir():
                if path.is_file():
                    archive.write(path, arcname=path.name)

        result["job_id"] = job_id
        result["download_endpoint"] = f"/results/{job_id}"
        result["disclaimer"] = "Research segmentation output; requires qualified clinical review."
        return result

    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid ZIP archive.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await file.close()


@app.get("/results/{job_id}")
def download_results(job_id: str):
    # job_id is generated as hex; this also prevents path traversal.
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    result_zip = OUTPUT_DIR / job_id / "segmentation_results.zip"
    if not result_zip.is_file():
        raise HTTPException(status_code=404, detail="Results not found.")
    return FileResponse(
        result_zip,
        media_type="application/zip",
        filename=f"{job_id}_segmentation_results.zip",
    )
