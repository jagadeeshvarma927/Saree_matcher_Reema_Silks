
import io
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.recommendation_service import RecommendationService


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

app = FastAPI(title="Saree Matcher")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the vision model once when the application starts.
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE)
model.eval()

# Load the catalog and embeddings once.
recommender = RecommendationService()


def create_embedding(image: Image.Image):
    """Generate an embedding using the same model as the catalog."""

    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        features = model.encode_image(tensor)
        features = features / features.norm(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-12)

    return features.squeeze(0).cpu().numpy()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"recommendations": None, "error": None}
    )

app.mount(
    "/catalog-images",
    StaticFiles(directory=str(BASE_DIR / "processed_images")),
    name="catalog-images"
)


@app.post("/", response_class=HTMLResponse)
async def find_sarees(
    request: Request,
    blouse_image: UploadFile = File(...),
    max_price: float = Form(3000),
    top_k: int = Form(5)
):
    if max_price < 0:
        raise HTTPException(
            status_code=400,
            detail="Maximum price cannot be negative."
        )

    if not 1 <= top_k <= 20:
        raise HTTPException(
            status_code=400,
            detail="Choose between 1 and 20 recommendations."
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if blouse_image.content_type not in allowed_types:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "recommendations": [],
                "error": "Please upload a JPG, PNG, or WEBP image.",
                "max_price": max_price,
                "top_k": top_k
            },
            status_code=400
        )

    try:
        contents = await blouse_image.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller."
            )

        with Image.open(io.BytesIO(contents)) as uploaded_image:
            uploaded_image.load()
            blouse = uploaded_image.convert("RGB")

        query_embedding = create_embedding(blouse)

        # The recommendation service needs a file path for color
        # comparison. Save the upload to a temporary location.
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        import uuid

        temp_path = upload_dir / f"{uuid.uuid4().hex}.jpg"
        blouse.save(temp_path, format="JPEG", quality=95)

        try:
            results = recommender.recommend(
                query_embedding=query_embedding,
                query_image_path=temp_path,
                top_k=top_k,
                max_price=max_price
            )
        finally:
            temp_path.unlink(missing_ok=True)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "recommendations": results,
                "error": None,
                "max_price": max_price,
                "top_k": top_k
            }
        )

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "recommendations": [],
                "error": f"Could not process the image: {error}",
                "max_price": max_price,
                "top_k": top_k
            },
            status_code=400
        )
