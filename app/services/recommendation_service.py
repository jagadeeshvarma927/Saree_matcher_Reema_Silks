"""Saree recommendation service with URL-only deduplication."""
import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
from PIL import Image, ImageOps

# Local CLIP is optional: the Hugging Face route can pass query_embedding.
try:
    import open_clip
    import torch
except ImportError:
    open_clip = None
    torch = None

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

_model = None
_preprocess = None
_catalog_cache = None
_embedding_cache = None
_embedding_ids_cache = None


def _load_catalog():
    global _catalog_cache
    if _catalog_cache is None:
        if not CATALOG_PATH.exists():
            raise FileNotFoundError(f"Catalog file not found: {CATALOG_PATH}")
        with CATALOG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("sarees.json must contain a JSON list.")
        _catalog_cache = data
    return _catalog_cache


def _load_embeddings():
    global _embedding_cache, _embedding_ids_cache
    if _embedding_cache is None:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Embedding file not found: {EMBEDDINGS_PATH}")
        if not EMBEDDING_IDS_PATH.exists():
            raise FileNotFoundError(f"Embedding ID file not found: {EMBEDDING_IDS_PATH}")
        _embedding_cache = np.load(EMBEDDINGS_PATH).astype(np.float32)
        with EMBEDDING_IDS_PATH.open("r", encoding="utf-8") as f:
            _embedding_ids_cache = json.load(f)
        if _embedding_cache.ndim != 2:
            raise ValueError("Embeddings must be a 2D NumPy array.")
        if len(_embedding_ids_cache) != len(_embedding_cache):
            raise ValueError("Embedding count does not match embedding ID count.")
        norms = np.linalg.norm(_embedding_cache, axis=1, keepdims=True)
        _embedding_cache /= np.maximum(norms, 1e-12)
    return _embedding_cache, _embedding_ids_cache


def _load_model():
    global _model, _preprocess, DEVICE
    if _model is None:
        if open_clip is None or torch is None:
            raise ImportError("Local image embedding requires torch and open_clip_torch.")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED_MODEL
        )
        _model = _model.to(DEVICE).eval()
    return _model, _preprocess


def _embed_query_image(image_path):
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
    candidates = []
    if item.get("image"):
        candidates.append(ORIGINAL_IMAGE_DIR / Path(item["image"]).name)
    if item.get("saree_id"):
        candidates.append(PROCESSED_IMAGE_DIR / f"{item['saree_id']}.jpg")
    for path in candidates:
        if path.exists():
            return path
    return None


def _canonical_product_url(url):
    """Canonicalize URLs for exact product URL deduplication only."""
    if not url:
        return None
    try:
        parsed = urlparse(str(url).strip())
        if not parsed.scheme or not parsed.netloc:
            return str(url).strip().rstrip("/").lower() or None
        host = (parsed.hostname or "").lower()
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunparse((parsed.scheme.lower(), host, parsed.path.rstrip("/"), "", "", ""))
    except Exception:
        return str(url).strip().rstrip("/").lower() or None


def _color_histogram(image_path):
    with Image.open(image_path) as im:
        image = ImageOps.exif_transpose(im).convert("HSV").resize((160, 160))
        hsv = np.asarray(image, dtype=np.uint8)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3), bins=(18, 4, 4),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    return hist / max(float(np.linalg.norm(hist)), 1e-12)


def _color_histogram_from_image(image):
    hsv = np.asarray(image.convert("HSV").resize((160, 160)), dtype=np.uint8)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3), bins=(18, 4, 4),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    return hist / max(float(np.linalg.norm(hist)), 1e-12)


def _color_similarity(query_hist, candidate_hist):
    return float(np.clip(np.dot(query_hist, candidate_hist), 0.0, 1.0))


def _visual_similarity_to_score(cosine_score):
    return float(np.clip((float(cosine_score) + 1.0) / 2.0, 0.0, 1.0))


def _normalize_weights(visual_weight, color_weight, use_color=True):
    total = float(visual_weight) + float(color_weight)
    if total <= 0:
        raise ValueError("visual_weight + color_weight must be greater than zero.")
    if not use_color:
        return 1.0, 0.0
    return float(visual_weight) / total, float(color_weight) / total


def _select_recommendations(scored, top_k):
    """Select top results, removing only duplicate canonical product URLs.

    No image-hash, perceptual-hash, visual-nearness, or title deduplication occurs.
    """
    selected, seen_urls = [], set()
    for candidate in scored:
        url_key = candidate.get("_canonical_url")
        if url_key and url_key in seen_urls:
            print(f"[DEDUP] Repeated product URL skipped: {candidate.get('name')}")
            continue
        if url_key:
            seen_urls.add(url_key)
        candidate.pop("_canonical_url", None)
        candidate.pop("_image_path", None)
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def get_recommendations(
    query_image_path, top_k=5, max_price=None,
    visual_weight=VISUAL_WEIGHT, color_weight=COLOR_WEIGHT,
    duplicate_distance=None,  # Backward-compatible argument; intentionally unused.
):
    """Recommend products using a query image embedded by local CLIP."""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    visual_weight, color_weight = _normalize_weights(visual_weight, color_weight)
    catalog = _load_catalog()
    embeddings, embedding_ids = _load_embeddings()
    query_embedding = _embed_query_image(query_image_path)
    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError("Query embedding dimension does not match catalog embeddings.")
    embedding_by_id = {str(sid): embeddings[i] for i, sid in enumerate(embedding_ids)}
    with Image.open(query_image_path) as im:
        query_hist = _color_histogram_from_image(ImageOps.exif_transpose(im).convert("RGB"))

    scored = []
    for item in catalog:
        vector = embedding_by_id.get(str(item.get("saree_id", "")))
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
        color_score = 0.0
        if image_path:
            try:
                color_score = _color_similarity(query_hist, _color_histogram(image_path))
            except Exception as exc:
                print(f"[WARN] Could not calculate color score for {image_path}: {exc}")
        visual_score = _visual_similarity_to_score(float(np.dot(query_embedding, vector)))
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
            "_canonical_url": _canonical_product_url(item.get("product_url")),
        })
        scored.append(result)
    scored.sort(key=lambda row: row["score"], reverse=True)
    return _select_recommendations(scored, top_k)


def get_recommendations_from_embedding(
    query_embedding, top_k=5, max_price=None,
    visual_weight=VISUAL_WEIGHT, color_weight=COLOR_WEIGHT,
    duplicate_distance=None,  # Backward-compatible argument; intentionally unused.
    query_image_path=None, **kwargs,
):
    """Rank products using an embedding supplied by app/main.py/Hugging Face."""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    catalog = _load_catalog()
    embeddings, embedding_ids = _load_embeddings()
    query_vector = np.squeeze(np.asarray(query_embedding, dtype=np.float32))
    if query_vector.ndim != 1:
        raise ValueError(f"query_embedding must be a vector; got shape {np.asarray(query_embedding).shape}")
    if query_vector.shape[0] != embeddings.shape[1]:
        raise ValueError("Query embedding dimension does not match catalog embeddings.")
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm <= 1e-12:
        raise ValueError("query_embedding has zero norm.")
    query_vector = query_vector / query_norm

    query_hist = None
    if query_image_path:
        try:
            with Image.open(query_image_path) as im:
                query_hist = _color_histogram_from_image(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception as exc:
            print(f"[WARN] Query color image unreadable; using visual score only: {exc}")

    embedding_by_id = {str(sid): embeddings[i] for i, sid in enumerate(embedding_ids)}
    visual_weight, color_weight = _normalize_weights(
        visual_weight, color_weight, use_color=query_hist is not None
    )
    scored = []
    for item in catalog:
        vector = embedding_by_id.get(str(item.get("saree_id", "")))
        if vector is None:
            continue
        price = item.get("price")
        if max_price is not None:
            try:
                if price is None or float(price) > float(max_price):
                    continue
            except (TypeError, ValueError):
                continue
        visual_score = _visual_similarity_to_score(float(np.dot(query_vector, vector)))
        image_path = _image_path_for_item(item)
        color_score = 0.0
        if query_hist is not None and image_path:
            try:
                color_score = _color_similarity(query_hist, _color_histogram(image_path))
            except Exception as exc:
                print(f"[WARN] Could not calculate color score for {image_path}: {exc}")
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
            "_canonical_url": _canonical_product_url(item.get("product_url")),
        })
        scored.append(result)
    scored.sort(key=lambda row: row["score"], reverse=True)
    return _select_recommendations(scored, top_k)


class RecommendationService:
    """Class interface expected by app/main.py."""
    def recommend(self, query_embedding=None, top_k=5, max_price=None,
                  query_image_path=None, **kwargs):
        if query_embedding is not None:
            return get_recommendations_from_embedding(
                query_embedding=query_embedding, top_k=top_k,
                max_price=max_price, query_image_path=query_image_path, **kwargs,
            )
        if query_image_path is not None:
            return get_recommendations(
                query_image_path=query_image_path, top_k=top_k,
                max_price=max_price, **kwargs,
            )
        raise ValueError("Provide query_embedding or query_image_path.")

    def get_recommendations(self, query_embedding=None, top_k=5, max_price=None,
                            query_image_path=None, **kwargs):
        return self.recommend(
            query_embedding=query_embedding, top_k=top_k, max_price=max_price,
            query_image_path=query_image_path, **kwargs,
        )


def recommend_sarees(query_image_path, top_k=5, max_price=None, **kwargs):
    """Backward-compatible function alias."""
    return get_recommendations(
        query_image_path=query_image_path, top_k=top_k,
        max_price=max_price, **kwargs,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test saree recommendations for an image.")
    parser.add_argument("image", help="Path to uploaded blouse-piece image")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-price", type=float, default=None)
    args = parser.parse_args()
    recommendations = get_recommendations(args.image, top_k=args.top_k, max_price=args.max_price)
    print("\n--- Saree recommendations ---")
    if not recommendations:
        print("No recommendations returned. Check catalog, embeddings, and image paths.")
    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. {item.get('name')} | ID={item.get('saree_id')} | "
            f"price={item.get('price')} | score={item.get('score')} | "
            f"visual={item.get('visual_similarity')} | color={item.get('color_similarity')}\n"
            f"   {item.get('product_url')}"
        )
