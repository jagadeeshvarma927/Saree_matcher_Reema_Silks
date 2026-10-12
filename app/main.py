
import io
import uuid
from pathlib import Path
from urllib.parse import urlparse

import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    ALLOWED_PRODUCT_DOMAINS,
    DEFAULT_MAX_PRICE,
    DEFAULT_TOP_K,
    DEVICE,
    MAX_TOP_K,
    MAX_UPLOAD_MB,
    MODEL_NAME,
    PRETRAINED_MODEL,
)
from app.services.recommendation_service import RecommendationService


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
PROCESSED_DIR = BASE_DIR / "processed_images"

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


def get_device():
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE=cuda, but CUDA is unavailable."
        )

    return DEVICE


ACTIVE_DEVICE = get_device()

# Load model once per worker process.
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED_MODEL,
)
model = model.to(ACTIVE_DEVICE)
model.eval()

# Load catalog embeddings once per worker process.
recommender = RecommendationService()


def create_embedding(image: Image.Image):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    tensor = tensor.to(ACTIVE_DEVICE)

    with torch.inference_mode():
        features = model.encode_image(tensor)
        features = features / features.norm(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-12)

    return features.squeeze(0).cpu().numpy()


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


def render_page(request, recommendations=None, error=None,
                max_price=DEFAULT_MAX_PRICE, top_k=DEFAULT_TOP_K,
                status_code=200):

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


@app.post("/", response_class=HTMLResponse)
async def find_sarees(
    request: Request,
    blouse_image: UploadFile = File(...),
    max_price: float = Form(DEFAULT_MAX_PRICE),
    top_k: int = Form(DEFAULT_TOP_K),
):
    if max_price < 0:
        return render_page(
            request, [], "Budget cannot be negative.",
            max_price, top_k, 400
        )

    if not 1 <= top_k <= MAX_TOP_K:
        return render_page(
            request, [],
            f"Choose between 1 and {MAX_TOP_K} recommendations.",
            max_price, top_k, 400
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if blouse_image.content_type not in allowed_types:
        return render_page(
            request, [],
            "Upload a JPG, PNG, or WEBP image.",
            max_price, top_k, 400
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    try:
        # Read at most one byte beyond the limit to detect oversized files.
        contents = await blouse_image.read(max_bytes + 1)

        if len(contents) > max_bytes:
            return render_page(
                request, [],
                f"Image must be {MAX_UPLOAD_MB} MB or smaller.",
                max_price, top_k, 413
            )

        with Image.open(io.BytesIO(contents)) as uploaded:
            uploaded.verify()

        with Image.open(io.BytesIO(contents)) as uploaded:
            uploaded.load()
            blouse = uploaded.convert("RGB")

        query_embedding = create_embedding(blouse)

        # The color service currently accepts a file path.
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        temp_path = upload_dir / f"{uuid.uuid4().hex}.jpg"

        try:
            blouse.save(temp_path, format="JPEG", quality=95)

            results = recommender.recommend(
                query_embedding=query_embedding,
                query_image_path=temp_path,
                top_k=top_k,
                max_price=max_price,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        return render_page(
            request, results, None, max_price, top_k
        )

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return render_page(
            request, [],
            f"Could not process the uploaded image: {error}",
            max_price, top_k, 400
        )
    finally:
        await blouse_image.close()
