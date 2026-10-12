
from pathlib import Path

from PIL import Image
import open_clip
import torch

from app.services.recommendation_service import RecommendationService


PROJECT_ROOT = Path(__file__).resolve().parent


def generate_embedding(image_path):
    """Generate a normalized OpenCLIP image embedding."""

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        features = model.encode_image(tensor)
        features = features / features.norm(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-12)

    return features.squeeze(0).cpu().numpy()


def main():
    # Temporary test: use the original S001 saree image.
    query_image = PROJECT_ROOT / "saree_images" / "saree001.jpg"

    query_embedding = generate_embedding(query_image)

    recommender = RecommendationService()

    results = recommender.recommend(
        query_embedding=query_embedding,
        query_image_path=query_image,
        top_k=5,
        max_price=3000
    )

    print("\n--- Top 5 saree recommendations ---\n")

    for rank, item in enumerate(results, start=1):
        print(f"Rank               : {rank}")
        print(f"Saree ID           : {item['saree_id']}")
        print(f"Name               : {item['name']}")
        print(f"Price              : {item['price']}")
        print(f"Visual similarity  : {item['visual_similarity']:.4f}")
        print(f"Color similarity   : {item['color_similarity']:.4f}")
        print(f"Final score        : {item['final_score']:.4f}")
        print(f"Image              : {item['image_path']}")
        print(f"Product URL        : {item['product_url']}")
        print("-" * 55)


if __name__ == "__main__":
    main()
