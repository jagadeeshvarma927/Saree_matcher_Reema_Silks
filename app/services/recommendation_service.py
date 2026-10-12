
import json
from pathlib import Path

from app.services.similarity_service import SimilarityService
from app.services.color_service import color_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = PROJECT_ROOT / "app" / "data" / "sarees.json"
ORIGINAL_IMAGE_DIR = PROJECT_ROOT / "saree_images"
PROCESSED_IMAGE_DIR = PROJECT_ROOT / "processed_images"


class RecommendationService:
    """Combine visual and color similarity to recommend sarees."""

    VISUAL_WEIGHT = 0.60
    COLOR_WEIGHT = 0.40

    def __init__(self):
        self.similarity_service = SimilarityService()

        with CATALOG_PATH.open("r", encoding="utf-8") as file:
            catalog = json.load(file)

        self.catalog_by_id = {
            item["saree_id"]: item
            for item in catalog
            if item.get("saree_id")
        }

        self.original_image_by_id = {
            item["saree_id"]: ORIGINAL_IMAGE_DIR / item["image"]
            for item in catalog
            if item.get("saree_id") and item.get("image")
        }

    def recommend(
        self,
        query_embedding,
        query_image_path,
        top_k=5,
        max_price=None
    ):
        """
        Rank catalog sarees using visual and color similarity.

        query_embedding:
            OpenCLIP embedding of the uploaded blouse image.

        query_image_path:
            Path to the original uploaded blouse image.

        top_k:
            Number of recommendations to return.

        max_price:
            Optional maximum price, in the same currency as catalog prices.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if max_price is not None and max_price < 0:
            raise ValueError("max_price cannot be negative.")

        # Retrieve all candidates, then apply business filters.
        candidates = self.similarity_service.search(
            query_embedding=query_embedding,
            top_k=len(self.similarity_service.saree_ids)
        )

        recommendations = []

        for candidate in candidates:
            saree_id = candidate["saree_id"]
            product = self.catalog_by_id[saree_id]

            price = product.get("price")

            # Exclude unknown prices when a price limit is requested.
            if max_price is not None:
                if price is None or float(price) > max_price:
                    continue

            saree_image_path = self.original_image_by_id.get(
                saree_id
            )

            if not saree_image_path or not saree_image_path.is_file():
                continue

            color_score = color_similarity(
                query_image_path,
                saree_image_path
            )

            visual_score = candidate["similarity"]

            # Convert cosine similarity from [-1, 1] to [0, 1].
            # This keeps the weighted score within a common range.
            visual_score_01 = max(
                0.0,
                min(1.0, (visual_score + 1.0) / 2.0)
            )

            final_score = (
                self.VISUAL_WEIGHT * visual_score_01
                + self.COLOR_WEIGHT * color_score
            )

            recommendations.append({
                "saree_id": saree_id,
                "name": product.get("name"),
                "price": price,
                "image": product.get("image"),
                "image_path": str(saree_image_path),
                "product_url": product.get("product_url"),
                "visual_similarity": round(
                    visual_score_01, 4
                ),
                "color_similarity": round(
                    color_score, 4
                ),
                "final_score": round(
                    final_score, 4
                )
            })

        recommendations.sort(
            key=lambda item: item["final_score"],
            reverse=True
        )

        return recommendations[:top_k]
