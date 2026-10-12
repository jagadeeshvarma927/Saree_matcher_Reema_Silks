
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.cluster import MiniBatchKMeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# IMAGE_DIR = PROJECT_ROOT / "processed_images"
IMAGE_DIR = PROJECT_ROOT / "saree_images"
N_COLORS = 5
MAX_PIXELS = 10000
RANDOM_STATE = 42


def rgb_to_lab(rgb_pixels):
    """Convert RGB pixels to CIELAB using Pillow."""

    image = Image.new("RGB", (len(rgb_pixels), 1))
    image.putdata([tuple(map(int, p)) for p in rgb_pixels])

    lab_image = image.convert("LAB")

    return np.asarray(lab_image, dtype=np.float32).reshape(
        -1, 3
    )


def extract_dominant_colors(image_path, n_colors=N_COLORS):
    """
    Extract representative colors from an image.

    Returns a dictionary containing representative RGB colors
    and their proportions.
    """

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((200, 200))
        pixels = np.asarray(image, dtype=np.uint8).reshape(
            -1, 3
        )

    if len(pixels) == 0:
        raise ValueError("Image contains no pixels.")

    # Sample pixels when the image is large.
    rng = np.random.default_rng(RANDOM_STATE)

    if len(pixels) > MAX_PIXELS:
        indices = rng.choice(
            len(pixels),
            MAX_PIXELS,
            replace=False
        )
        pixels = pixels[indices]

    actual_k = min(n_colors, len(pixels))

    model = MiniBatchKMeans(
        n_clusters=actual_k,
        random_state=RANDOM_STATE,
        n_init=3,
        batch_size=1024
    )

    labels = model.fit_predict(pixels)

    counts = np.bincount(
        labels,
        minlength=actual_k
    )

    proportions = counts / counts.sum()

    # Sort dominant colors from most frequent to least frequent.
    order = np.argsort(proportions)[::-1]

    colors = []

    for index in order:
        center = np.clip(
            model.cluster_centers_[index],
            0,
            255
        ).astype(int)

        colors.append({
            "rgb": center.tolist(),
            "proportion": round(
                float(proportions[index]), 4
            )
        })

    return {
        "colors": colors
    }


def color_similarity(image_a, image_b):
    """
    Compare two images using their dominant color palettes.

    Returns a score between 0 and 1.
    """

    palette_a = extract_dominant_colors(image_a)["colors"]
    palette_b = extract_dominant_colors(image_b)["colors"]

    rgb_a = np.array(
        [item["rgb"] for item in palette_a],
        dtype=np.uint8
    )

    rgb_b = np.array(
        [item["rgb"] for item in palette_b],
        dtype=np.uint8
    )

    weights_a = np.array(
        [item["proportion"] for item in palette_a]
    )

    weights_b = np.array(
        [item["proportion"] for item in palette_b]
    )

    lab_a = rgb_to_lab(rgb_a)
    lab_b = rgb_to_lab(rgb_b)

    distances = pairwise_distances(
        lab_a,
        lab_b,
        metric="euclidean"
    )

    # Convert perceptual distance into a soft similarity.
    # This is an initial heuristic, to be tuned using examples.
    similarities = np.exp(-distances / 35.0)

    # Each color in A gets the similarity of its closest
    # color in B, weighted by its proportion.
    best_a_to_b = similarities.max(axis=1)
    score_a_to_b = np.sum(weights_a * best_a_to_b)

    best_b_to_a = similarities.max(axis=0)
    score_b_to_a = np.sum(weights_b * best_b_to_a)

    return float(
        (score_a_to_b + score_b_to_a) / 2.0
    )



if __name__ == "__main__":
    import json

    with open(
        PROJECT_ROOT / "app" / "data" / "sarees.json",
        "r",
        encoding="utf-8"
    ) as file:
        catalog = json.load(file)

    catalog_by_id = {
        item["saree_id"]: item
        for item in catalog
    }

    blouse_item = catalog_by_id["S001"]
    saree_item = catalog_by_id["S020"]

    blouse_path = IMAGE_DIR / blouse_item["image"]
    saree_path = IMAGE_DIR / saree_item["image"]

    print("Blouse palette:")
    print(extract_dominant_colors(blouse_path))

    print("\nSaree palette:")
    print(extract_dominant_colors(saree_path))

    score = color_similarity(blouse_path, saree_path)

    print(f"\nColor similarity: {score:.4f}")
