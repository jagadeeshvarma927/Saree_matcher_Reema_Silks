"""
Duplicate-aware saree recommendation service.

Expected project layout:
    app/data/sarees.json
    app/data/saree_embeddings.npy
    app/data/saree_embedding_ids.json
    processed_images/S00001.jpg
    saree_images/saree00001.jpg

This module:
- Loads precomputed CLIP embeddings and maps them by saree_id.
- Scores candidates using image-embedding similarity and a color histogram.
- Filters duplicate product URLs and visually near-identical catalog images.
- Returns distinct recommendations in descending score order.

Install:
    pip install numpy Pillow open_clip_torch
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
from PIL import Image, ImageOps

try:
    import open_clip
    import torch
except ImportError:
    open_clip = None
    torch = None


# ---------------------------------------------------------------------
# Paths and ranking configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "app" / "data"
CATALOG_PATH = DATA_DIR / "sarees.json"
EMBEDDINGS_PATH = DATA_DIR / "saree_embeddings.npy"
EMBEDDING_IDS_PATH = DATA_DIR / "saree_embedding_ids.json"
ORIGINAL_IMAGE_DIR = BASE_DIR / "saree_images"
PROCESSED_IMAGE_DIR = BASE_DIR / "processed_images"

MODEL_NAME = "ViT-B-32"
PRETRAINED_MODEL = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

VISUAL_WEIGHT = 0.70
COLOR_WEIGHT = 0.30

# Hamming distance on a 64-bit perceptual hash. Lower means more alike.
# <= 5 is a conservative default for repeated/near-identical product photos.
PHASH_DUPLICATE_DISTANCE = 5

_model = None
_preprocess = None
_catalog_cache = None
_embedding_cache = None
_embedding_ids_cache = None
_hash_cache = {}


def _load_catalog():
    global _catalog_cache
    if _catalog_cache is None:
        if not CATALOG_PATH.exists():
            raise FileNotFoundError(f"Catalog file not found: {CATALOG_PATH}")
        with CATALOG_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("sarees.json must contain a JSON list.")
        _catalog_cache = data
    return _catalog_cache


def _load_embeddings():
    global _embedding_cache, _embedding_ids_cache

    if _embedding_cache is None:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {EMBEDDINGS_PATH}. "
                "Run the embedding generation script after updating the catalog."
            )
        if not EMBEDDING_IDS_PATH.exists():
            raise FileNotFoundError(
                f"Embedding ID file not found: {EMBEDDING_IDS_PATH}"
            )

        _embedding_cache = np.load(EMBEDDINGS_PATH).astype(np.float32)
        with EMBEDDING_IDS_PATH.open("r", encoding="utf-8") as file:
            _embedding_ids_cache = json.load(file)

        if _embedding_cache.ndim != 2:
            raise ValueError("Embeddings must be a 2D NumPy array.")
        if len(_embedding_ids_cache) != len(_embedding_cache):
            raise ValueError(
                "Embedding count does not match embedding ID count. "
                "Regenerate embeddings from the current catalog."
            )

        # Normalize rows to unit vectors for cosine similarity.
        norms = np.linalg.norm(_embedding_cache, axis=1, keepdims=True)
        _embedding_cache = _embedding_cache / np.maximum(norms, 1e-12)

    return _embedding_cache, _embedding_ids_cache


def _load_model():
    global _model, _preprocess
    if _model is None:
        if open_clip is None or torch is None:
            raise ImportError(
                "open_clip_torch and torch are required to embed the uploaded image. "
                "Install with: pip install open_clip_torch"
            )
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED_MODEL
        )
        _model = _model.to(DEVICE).eval()
    return _model, _preprocess


def _embed_query_image(image_path):
    """Create a normalized CLIP embedding for the uploaded blouse-piece image."""
    model, preprocess = _load_model()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Uploaded image not found: {image_path}")

    with Image.open(image_path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(tensor).float()
        embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    return embedding.cpu().numpy()[0].astype(np.float32)


def _image_path_for_item(item):
    """Resolve the original catalog image, then fall back to processed image."""
    image_name = item.get("image")
    saree_id = item.get("saree_id")

    candidates = []
    if image_name:
        candidates.append(ORIGINAL_IMAGE_DIR / Path(image_name).name)
    if saree_id:
        candidates.append(PROCESSED_IMAGE_DIR / f"{saree_id}.jpg")

    for path in candidates:
        if path.exists():
            return path
    return None


def _canonical_product_url(url):
    """Remove query strings, fragments and trailing slashes from product URLs."""
    if not url:
        return None
    try:
        parsed = urlparse(str(url).strip())
        if not parsed.scheme or not parsed.netloc:
            return str(url).strip().rstrip("/").lower() or None
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/")
        return urlunparse((parsed.scheme.lower(), host, path, "", "", ""))
    except Exception:
        return str(url).strip().rstrip("/").lower() or None


def _phash(image_path):
    """
    Return a simple 64-bit perceptual hash without an extra dependency.
    This is a DCT-free 8x8 average hash, useful for identical/similar photos.
    """
    key = str(image_path)
    if key in _hash_cache:
        return _hash_cache[key]

    with Image.open(image_path) as im:
        gray = ImageOps.exif_transpose(im).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)

    mean = float(pixels.mean())
    bits = (pixels >= mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    _hash_cache[key] = value
    return value


def _hash_distance(hash_a, hash_b):
    return (int(hash_a) ^ int(hash_b)).bit_count()


def _color_histogram(image_path):
    """Return a normalized HSV histogram from an original image."""
    with Image.open(image_path) as im:
        hsv = np.asarray(
            ImageOps.exif_transpose(im).convert("HSV").resize((160, 160)),
            dtype=np.uint8,
        )

    # Histogram hue and saturation/value together; normalize for image size.
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=(18, 4, 4),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    norm = np.linalg.norm(hist)
    return hist / max(float(norm), 1e-12)


def _color_similarity(query_hist, candidate_hist):
    """Cosine similarity between normalized color histograms, clipped to [0, 1]."""
    return float(np.clip(np.dot(query_hist, candidate_hist), 0.0, 1.0))


def _visual_similarity_to_score(cosine_score):
    """Map CLIP cosine similarity to a user-friendly [0, 1] score."""
    return float(np.clip((float(cosine_score) + 1.0) / 2.0, 0.0, 1.0))


def _dedupe_key_name(name):
    """Normalize title for reporting; titles alone are not used to discard products."""
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def get_recommendations(
    query_image_path,
    top_k=5,
    max_price=None,
    visual_weight=VISUAL_WEIGHT,
    color_weight=COLOR_WEIGHT,
    duplicate_distance=PHASH_DUPLICATE_DISTANCE,
):
    """
    Return distinct saree recommendations.

    Each returned dictionary includes:
      saree_id, name, price, image, image_url, product_url,
      visual_similarity, color_similarity, score

    Duplicate filtering is greedy: the highest-ranked candidate is kept,
    and lower-ranked candidates with the same canonical URL or near-identical
    image are skipped.
    """
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    total_weight = float(visual_weight) + float(color_weight)
    if total_weight <= 0:
        raise ValueError("visual_weight + color_weight must be greater than zero.")
    visual_weight = float(visual_weight) / total_weight
    color_weight = float(color_weight) / total_weight

    catalog = _load_catalog()
    embeddings, embedding_ids = _load_embeddings()
    query_embedding = _embed_query_image(query_image_path)

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            f"Query embedding size {query_embedding.shape[0]} does not match "
            f"catalog embedding size {embeddings.shape[1]}. Regenerate embeddings "
            "using the same CLIP model."
        )

    # Map IDs to their vectors, instead of assuming catalog and embedding
    # rows are in identical order.
    embedding_by_id = {
        str(saree_id): embeddings[i]
        for i, saree_id in enumerate(embedding_ids)
    }

    with Image.open(query_image_path) as im:
        query_image = ImageOps.exif_transpose(im).convert("RGB")
        query_hist = _color_histogram_from_image(query_image)

    scored = []
    for item in catalog:
        saree_id = str(item.get("saree_id", ""))
        vector = embedding_by_id.get(saree_id)
        if vector is None:
            continue

        price = item.get("price")
        if max_price is not None and price is not None:
            try:
                if float(price) > float(max_price):
                    continue
            except (TypeError, ValueError):
                pass

        image_path = _image_path_for_item(item)
        if image_path is None:
            # Visual recommendations can still be ranked, but duplicate-image
            # filtering and color scoring need an available local image.
            color_score = 0.0
            image_hash = None
        else:
            try:
                candidate_hist = _color_histogram(image_path)
                color_score = _color_similarity(query_hist, candidate_hist)
                image_hash = _phash(image_path)
            except Exception as error:
                print(f"[WARN] Could not inspect {image_path}: {error}")
                color_score = 0.0
                image_hash = None

        cosine = float(np.dot(query_embedding, vector))
        visual_score = _visual_similarity_to_score(cosine)
        final_score = visual_weight * visual_score + color_weight * color_score

        result = dict(item)
        result.update({
            "visual_similarity": round(visual_score, 4),
            "color_similarity": round(color_score, 4),
            "visual_score": round(visual_score, 4),
            "color_score": round(color_score, 4),
            "score": round(float(final_score), 4),
            "final_score": round(float(final_score), 4),
            "_image_path": str(image_path) if image_path else None,
            "_image_hash": image_hash,
            "_canonical_url": _canonical_product_url(item.get("product_url")),
        })
        scored.append(result)

    scored.sort(key=lambda row: row["score"], reverse=True)

    # Greedy duplicate removal: keep the best-scoring candidate first.
    selected = []
    seen_urls = set()
    seen_hashes = []

    for candidate in scored:
        url_key = candidate.get("_canonical_url")
        if url_key and url_key in seen_urls:
            print(f"[DEDUP] Repeated product URL skipped: {candidate.get('name')}")
            continue

        image_hash = candidate.get("_image_hash")
        if image_hash is not None:
            duplicate = any(
                _hash_distance(image_hash, existing_hash) <= duplicate_distance
                for existing_hash in seen_hashes
            )
            if duplicate:
                print(f"[DEDUP] Near-identical image skipped: {candidate.get('name')}")
                continue

        # Do not deduplicate by product title alone: many real products share
        # generic titles such as "Banarasi Georgette Saree".
        if url_key:
            seen_urls.add(url_key)
        if image_hash is not None:
            seen_hashes.append(image_hash)

        candidate.pop("_image_path", None)
        candidate.pop("_image_hash", None)
        candidate.pop("_canonical_url", None)
        selected.append(candidate)

        if len(selected) >= top_k:
            break

    return selected



def get_recommendations_from_embedding(
    query_embedding,
    top_k=5,
    max_price=None,
    visual_weight=VISUAL_WEIGHT,
    color_weight=COLOR_WEIGHT,
    duplicate_distance=PHASH_DUPLICATE_DISTANCE,
    query_image_path=None,
    **kwargs,
):
    """
    Rank catalog products using an embedding already generated by app/main.py.

    This avoids re-embedding the uploaded image. If query_image_path is supplied,
    color similarity is included; otherwise ranking uses visual similarity only.
    """
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    catalog = _load_catalog()
    embeddings, embedding_ids = _load_embeddings()

    query_vector = np.asarray(query_embedding, dtype=np.float32)
    query_vector = np.squeeze(query_vector)
    if query_vector.ndim != 1:
        raise ValueError(
            f"query_embedding must be a 1D vector or a single-row array; "
            f"received shape {np.asarray(query_embedding).shape}"
        )

    if query_vector.shape[0] != embeddings.shape[1]:
        raise ValueError(
            f"Query embedding size {query_vector.shape[0]} does not match "
            f"catalog embedding size {embeddings.shape[1]}. Ensure main.py "
            f"uses the same CLIP model as the catalog embedding generator."
        )

    query_norm = float(np.linalg.norm(query_vector))
    if query_norm <= 1e-12:
        raise ValueError("query_embedding has zero norm.")
    query_vector = query_vector / query_norm

    query_hist = None
    if query_image_path:
        try:
            with Image.open(query_image_path) as im:
                query_image = ImageOps.exif_transpose(im).convert("RGB")
                query_hist = _color_histogram_from_image(query_image)
        except Exception as error:
            print(f"[WARN] Query color image could not be read; using visual score only: {error}")

    embedding_by_id = {
        str(saree_id): embeddings[i]
        for i, saree_id in enumerate(embedding_ids)
    }

    total_weight = float(visual_weight) + float(color_weight)
    if total_weight <= 0:
        raise ValueError("visual_weight + color_weight must be greater than zero.")

    # If there is no query image for color comparison, use visual similarity alone.
    if query_hist is None:
        visual_weight, color_weight = 1.0, 0.0
    else:
        visual_weight = float(visual_weight) / total_weight
        color_weight = float(color_weight) / total_weight

    scored = []
    for item in catalog:
        saree_id = str(item.get("saree_id", ""))
        vector = embedding_by_id.get(saree_id)
        if vector is None:
            continue

        price = item.get("price")
        if max_price is not None:
            # Exclude unknown prices when a maximum-price filter is requested.
            try:
                if price is None or float(price) > float(max_price):
                    continue
            except (TypeError, ValueError):
                continue

        cosine = float(np.dot(query_vector, vector))
        visual_score = _visual_similarity_to_score(cosine)

        image_path = _image_path_for_item(item)
        image_hash = None
        color_score = 0.0
        if image_path:
            try:
                image_hash = _phash(image_path)
                if query_hist is not None:
                    color_score = _color_similarity(
                        query_hist, _color_histogram(image_path)
                    )
            except Exception as error:
                print(f"[WARN] Could not inspect {image_path}: {error}")

        final_score = visual_weight * visual_score + color_weight * color_score
        result = dict(item)
        result.update({
            "visual_similarity": round(visual_score, 4),
            "color_similarity": round(color_score, 4),
            "visual_score": round(visual_score, 4),
            "color_score": round(color_score, 4),
            "score": round(float(final_score), 4),
            "final_score": round(float(final_score), 4),
            "_image_hash": image_hash,
            "_canonical_url": _canonical_product_url(item.get("product_url")),
        })
        scored.append(result)

    scored.sort(key=lambda row: row["score"], reverse=True)

    selected = []
    seen_urls = set()
    seen_hashes = []
    for candidate in scored:
        url_key = candidate.get("_canonical_url")
        if url_key and url_key in seen_urls:
            print(f"[DEDUP] Repeated product URL skipped: {candidate.get('name')}")
            continue

        image_hash = candidate.get("_image_hash")
        if image_hash is not None and any(
            _hash_distance(image_hash, prior_hash) <= duplicate_distance
            for prior_hash in seen_hashes
        ):
            print(f"[DEDUP] Near-identical image skipped: {candidate.get('name')}")
            continue

        if url_key:
            seen_urls.add(url_key)
        if image_hash is not None:
            seen_hashes.append(image_hash)

        candidate.pop("_image_hash", None)
        candidate.pop("_canonical_url", None)
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    return selected


class RecommendationService:
    """Class interface expected by app/main.py."""

    def recommend(
        self,
        query_embedding=None,
        top_k=5,
        max_price=None,
        query_image_path=None,
        **kwargs,
    ):
        if query_embedding is not None:
            return get_recommendations_from_embedding(
                query_embedding=query_embedding,
                top_k=top_k,
                max_price=max_price,
                query_image_path=query_image_path,
                **kwargs,
            )

        if query_image_path is not None:
            return get_recommendations(
                query_image_path=query_image_path,
                top_k=top_k,
                max_price=max_price,
                **kwargs,
            )

        raise ValueError("Provide query_embedding or query_image_path.")

    def get_recommendations(
        self, query_embedding=None, top_k=5, max_price=None,
        query_image_path=None, **kwargs
    ):
        return self.recommend(
            query_embedding=query_embedding,
            top_k=top_k,
            max_price=max_price,
            query_image_path=query_image_path,
            **kwargs,
        )


def _color_histogram_from_image(image):
    """Histogram helper for a PIL image object."""
    hsv = np.asarray(image.convert("HSV").resize((160, 160)), dtype=np.uint8)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=(18, 4, 4),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    norm = np.linalg.norm(hist)
    return hist / max(float(norm), 1e-12)


# Backwards-compatible alias for code that imports recommend_sarees.
def recommend_sarees(query_image_path, top_k=5, max_price=None, **kwargs):
    return get_recommendations(
        query_image_path=query_image_path,
        top_k=top_k,
        max_price=max_price,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test distinct saree recommendations for a query image."
    )
    parser.add_argument("image", help="Path to a blouse-piece/query image")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-price", type=float, default=None)
    args = parser.parse_args()

    recommendations = get_recommendations(
        args.image,
        top_k=args.top_k,
        max_price=args.max_price,
    )

    print("\n--- Distinct recommendations ---")
    if not recommendations:
        print("No recommendations returned. Check catalog, embeddings and image paths.")
    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. {item.get('name')} | ID={item.get('saree_id')} | "
            f"price={item.get('price')} | score={item.get('score')} | "
            f"visual={item.get('visual_similarity')} | "
            f"color={item.get('color_similarity')}\n"
            f"   {item.get('product_url')}"
        )
