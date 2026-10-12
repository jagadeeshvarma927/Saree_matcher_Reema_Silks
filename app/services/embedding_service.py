
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = PROJECT_ROOT / "app" / "data" / "sarees.json"
IMAGE_DIR = PROJECT_ROOT / "processed_images"

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "app" / "data" / "saree_embeddings.npy"
)

IDS_PATH = (
    PROJECT_ROOT / "app" / "data" / "saree_embedding_ids.json"
)

# Pretrained vision model; no custom training required
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_catalog():
    """Load the existing saree catalog."""

    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    if not isinstance(catalog, list):
        raise ValueError("sarees.json must contain a JSON list.")

    return catalog


def load_model():
    """Load OpenCLIP and select the available device."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    model = model.to(device)
    model.eval()

    return model, preprocess, device


def generate_embeddings():
    """Generate and save one normalized embedding per saree."""

    catalog = load_catalog()

    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model, preprocess, device = load_model()

    embeddings = []
    embedding_ids = []

    for index, item in enumerate(catalog, start=1):

        saree_id = item.get("saree_id")

        if not saree_id:
            print(f"[SKIP] Record {index}: missing saree_id")
            continue

        image_path = IMAGE_DIR / f"{saree_id}.jpg"

        if not image_path.is_file():
            print(f"[SKIP] {saree_id}: processed image not found")
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0)

            image_tensor = image_tensor.to(device)

            with torch.inference_mode():
                features = model.encode_image(image_tensor)

                # Normalize so cosine similarity can be
                # calculated efficiently using dot products.
                features = features / features.norm(
                    dim=-1,
                    keepdim=True
                ).clamp(min=1e-12)

            vector = features.squeeze(0).cpu().numpy()
            vector = vector.astype(np.float32)

            embeddings.append(vector)
            embedding_ids.append(saree_id)

            print(
                f"[{index}/{len(catalog)}] "
                f"Embedded {saree_id}"
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
            UnidentifiedImageError
        ) as error:
            print(f"[ERROR] {saree_id}: {error}")

    if not embeddings:
        raise RuntimeError(
            "No embeddings generated. Check the processed images."
        )

    embedding_matrix = np.stack(embeddings).astype(np.float32)

    # Save the numerical vectors.
    np.save(EMBEDDINGS_PATH, embedding_matrix)

    # Save the IDs in the exact same order as the vectors.
    with IDS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            embedding_ids,
            file,
            indent=4
        )

    print("\n--- Embedding generation completed ---")
    print(f"Catalog records : {len(catalog)}")
    print(f"Embeddings saved: {len(embedding_ids)}")
    print(f"Embedding shape : {embedding_matrix.shape}")
    print(f"Vector file     : {EMBEDDINGS_PATH}")
    print(f"ID mapping      : {IDS_PATH}")


if __name__ == "__main__":
    generate_embeddings()
