
import io
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gradio_client import Client, handle_file

from app.config import (
    ALLOWED_PRODUCT_DOMAINS,
    DEFAULT_MAX_PRICE,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MAX_UPLOAD_MB,
)
from app.services.recommendation_service import RecommendationService


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
PROCESSED_DIR = BASE_DIR / "processed_images"
UPLOAD_DIR = BASE_DIR / "uploads"

HF_SPACE_ID = os.getenv("HF_SPACE_ID", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

app = FastAPI(title="Saree Matcher")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if not PROCESSED_DIR.is_dir():
    raise RuntimeError(
        f"Processed image directory not found: {PROCESSED_DIR}"
    )

app.mount(
    "/catalog-images",
    StaticFiles(directory=str(PROCESSED_DIR)),
    name="catalog-images",
)

# Create the recommendation service without loading OpenCLIP.
recommender = RecommendationService()

# Reuse the Hugging Face client instead of creating one per request.
_hf_client = None


def get_hf_client():
    """Create the Hugging Face Space client on first use."""
    global _hf_client

    if _hf_client is None:
        if not HF_SPACE_ID:
            raise RuntimeError(
                "HF_SPACE_ID is not configured. "
                "Set it to your Hugging Face Space ID, "
                "for example username/saree-openclip-api."
            )

        client_args = {"token": HF_TOKEN} if HF_TOKEN else {}
        _hf_client = Client(HF_SPACE_ID, **client_args)

    return _hf_client



def create_remote_embedding(image_path: Path) -> np.ndarray:
    """Generate an embedding using the Hugging Face Space."""
    print(f"[HF] Connecting to Space: {HF_SPACE_ID}")

    client = get_hf_client()
    print("[HF] Client initialized; requesting image embedding...")

    result = client.predict(
        handle_file(str(image_path)),
        api_name="/embed",
    )

    print(f"[HF] Response type: {type(result).__name__}")

    if not isinstance(result, dict):
        raise ValueError(
            f"Unexpected Hugging Face response: {type(result).__name__}"
        )

    embedding = np.asarray(
        result.get("embedding"),
        dtype=np.float32,
    )

    print(f"[HF] Model: {result.get('model')}")
    print(f"[HF] Returned dimension: {embedding.shape}")

    if embedding.shape != (512,):
        raise ValueError(
            f"Expected embedding shape (512,), got {embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains invalid numeric values.")

    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError("Embedding is zero.")

    print("[HF] Embedding received and validated successfully.")
    return embedding / norm


def product_url_is_allowed(url):
    if not url:
        return False

    parsed = urlparse(url)

    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.hostname.lower() in ALLOWED_PRODUCT_DOMAINS
        and parsed.username is None
        and parsed.password is None
    )


def render_page(
    request,
    recommendations=None,
    error=None,
    max_price=DEFAULT_MAX_PRICE,
    top_k=DEFAULT_TOP_K,
    status_code=200,
):
    safe_recommendations = None

    if recommendations is not None:
        safe_recommendations = []

        for item in recommendations:
            result = dict(item)
            result["product_url_allowed"] = product_url_is_allowed(
                item.get("product_url")
            )
            safe_recommendations.append(result)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "recommendations": safe_recommendations,
            "error": error,
            "max_price": max_price,
            "top_k": top_k,
        },
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_page(request)


@app.get("/health")
async def health():
    """Basic health check; does not call or load the CLIP model."""
    return {
        "status": "ok",
        "inference": "huggingface-space",
        "space_configured": bool(HF_SPACE_ID),
    }


@app.post("/", response_class=HTMLResponse)
async def find_sarees(
    request: Request,
    blouse_image: UploadFile = File(...),
    max_price: float = Form(DEFAULT_MAX_PRICE),
    top_k: int = Form(DEFAULT_TOP_K),
):
    if max_price < 0:
        return render_page(
            request,
            [],
            "Budget cannot be negative.",
            max_price,
            top_k,
            400,
        )

    if not 1 <= top_k <= MAX_TOP_K:
        return render_page(
            request,
            [],
            f"Choose between 1 and {MAX_TOP_K} recommendations.",
            max_price,
            top_k,
            400,
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if blouse_image.content_type not in allowed_types:
        return render_page(
            request,
            [],
            "Upload a JPG, PNG, or WEBP image.",
            max_price,
            top_k,
            400,
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    try:
        contents = await blouse_image.read(max_bytes + 1)

        if len(contents) > max_bytes:
            return render_page(
                request,
                [],
                f"Image must be {MAX_UPLOAD_MB} MB or smaller.",
                max_price,
                top_k,
                413,
            )

        with Image.open(io.BytesIO(contents)) as uploaded:
            uploaded.verify()

        with Image.open(io.BytesIO(contents)) as uploaded:
            uploaded.load()
            blouse = uploaded.convert("RGB")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"

        try:
            # Save one normalized image for both remote inference
            # and the existing color-similarity recommender.
            blouse.save(temp_path, format="JPEG", quality=95)

            # Heavy OpenCLIP inference runs on Hugging Face, not Render.
            query_embedding = create_remote_embedding(temp_path)

            # Keep the existing local catalog matching and ranking logic.
            results = recommender.recommend(
                query_embedding=query_embedding,
                query_image_path=temp_path,
                top_k=top_k,
                max_price=max_price,
            )

            return render_page(
                request,
                results,
                None,
                max_price,
                top_k,
            )

        finally:
            temp_path.unlink(missing_ok=True)

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return render_page(
            request,
            [],
            f"Could not process the uploaded image: {error}",
            max_price,
            top_k,
            400,
        )

    except Exception as error:
        # Return a useful page instead of exposing an unhandled server error.
        print(f"[ERROR] Saree recommendation failed: {error}")
        return render_page(
            request,
            [],
            "Image recommendation failed. The Hugging Face Space may be "
            "starting up, unavailable, or incorrectly configured. "
            "Please try again shortly.",
            max_price,
            top_k,
            502,
        )

    finally:
        await blouse_image.close()
