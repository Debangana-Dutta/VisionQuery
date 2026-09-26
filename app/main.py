"""
app/main.py
-----------
FastAPI backend for VisionQuery.

The BLIP model is loaded lazily on the first /predict request.
This keeps API startup fast and avoids loading the 1.45 GB model
until it is actually needed.
"""

import os
import sys
from io import BytesIO

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)


from src.model import load_model
from src.inference import answer_question


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "configs",
    "config.yaml",
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

checkpoint_dir = os.path.join(
    BASE_DIR,
    cfg["training"]["output_dir"],
)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VisionQuery API",
    description="Visual Question Answering API powered by fine-tuned BLIP.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazy model state
# ---------------------------------------------------------------------------

processor = None
model = None
device = None
used_finetuned = False


def get_model():
    """
    Load the BLIP model the first time it is needed.

    Subsequent requests reuse the already-loaded model.
    """

    global processor, model, device, used_finetuned

    if model is None:
        print("[VisionQuery] Loading BLIP model...")
        print(f"[VisionQuery] Checkpoint: {checkpoint_dir}")

        processor, model, device, used_finetuned = load_model(
            model_name=cfg["model"]["name"],
            checkpoint_dir=checkpoint_dir,
        )

        print(f"[VisionQuery] Device: {device}")
        print(
            f"[VisionQuery] Fine-tuned checkpoint: "
            f"{used_finetuned}"
        )
        print("[VisionQuery] Model loaded successfully.")

    return processor, model, device, used_finetuned


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "name": "VisionQuery API",
        "status": "running",
        "model": cfg["model"]["name"],
        "fine_tuned": used_finetuned,
        "model_loaded": model is not None,
        "device": str(device) if device else "not loaded",
    }


@app.get("/health")
def health():
    """
    Health-check endpoint.
    """

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "fine_tuned": used_finetuned,
        "device": str(device) if device else "not loaded",
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    """
    Accept an image and natural-language question,
    then return the BLIP VQA prediction.
    """

    # -----------------------------------------------------------------------
    # Validate question
    # -----------------------------------------------------------------------

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------------------------------------------------
    # Validate uploaded file
    # -----------------------------------------------------------------------

    if (
        image.content_type
        and not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        # -------------------------------------------------------------------
        # Read image
        # -------------------------------------------------------------------

        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        pil_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # -------------------------------------------------------------------
        # Load model lazily
        # -------------------------------------------------------------------

        processor, model, device, used_finetuned = get_model()

        # -------------------------------------------------------------------
        # Run VQA inference
        # -------------------------------------------------------------------

        answer = answer_question(
            pil_image,
            question,
            processor,
            model,
            device,
            max_answer_length=cfg["model"]["max_answer_length"],
        )

        # -------------------------------------------------------------------
        # Return prediction
        # -------------------------------------------------------------------

        return {
            "answer": answer,
            "question": question,
            "model": cfg["model"]["name"],
            "fine_tuned": used_finetuned,
            "device": str(device),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )