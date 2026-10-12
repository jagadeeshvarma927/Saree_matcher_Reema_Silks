
from pathlib import Path

from PIL import Image
import open_clip
import torch

from app.services.similarity_service import SimilarityService


PROJECT_ROOT = Path(__file__).resolve().parent


def generate_image_embedding(image_path):
    """Encode an image using the same OpenCLIP model."""

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
    # Use S001 as a test query.
    query_image = (
        PROJECT_ROOT / "processed_images" / "S001.jpg"
    )

    query_embedding = generate_image_embedding(query_image)

    service = SimilarityService()

    results = service.search(
        query_embedding=query_embedding,
        top_k=5
    )

    print("\n--- Top 5 similar sarees ---\n")

    for rank, item in enumerate(results, start=1):
        print(f"Rank          : {rank}")
        print(f"Saree ID      : {item['saree_id']}")
        print(f"Name          : {item['name']}")
        print(f"Price         : {item['price']}")
        print(f"Similarity    : {item['similarity']:.4f}")
        print(f"Image         : {item['image_path']}")
        print(f"Product URL   : {item['product_url']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
