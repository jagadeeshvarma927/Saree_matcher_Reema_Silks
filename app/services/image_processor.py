
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = PROJECT_ROOT / "app" / "data" / "sarees.json"
SOURCE_DIR = PROJECT_ROOT / "saree_images"
OUTPUT_DIR = PROJECT_ROOT / "processed_images"
REPORT_PATH = PROJECT_ROOT / "processing_report.json"

# Consistent dimensions for previews and downstream processing
IMAGE_SIZE = (512, 512)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_catalog():
    """Load saree metadata from the JSON catalog."""

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG_PATH}"
        )

    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    if not isinstance(catalog, list):
        raise ValueError("sarees.json must contain a JSON list.")

    return catalog


def preprocess_image(source_path, output_path):
    """
    Correct image orientation, convert to RGB, and resize
    while preserving the complete image and its aspect ratio.
    """

    with Image.open(source_path) as image:
        # Respect orientation metadata from phone photographs
        image = ImageOps.exif_transpose(image)

        # JPEG output requires RGB; preserve image appearance
        image = image.convert("RGB")

        # Fit inside a square without stretching or cropping
        image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Place the image on a neutral canvas
        canvas = Image.new("RGB", IMAGE_SIZE, (245, 245, 245))

        x = (IMAGE_SIZE[0] - image.width) // 2
        y = (IMAGE_SIZE[1] - image.height) // 2

        canvas.paste(image, (x, y))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        canvas.save(
            output_path,
            format="JPEG",
            quality=92,
            optimize=True
        )


def process_catalog():
    """Process every catalog image and generate a report."""

    catalog = load_catalog()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "total_catalog_records": len(catalog),
        "processed": 0,
        "failed": 0,
        "missing": 0,
        "results": []
    }

    for item in catalog:
        saree_id = item.get("saree_id", "UNKNOWN")
        filename = item.get("image")

        if not filename:
            report["failed"] += 1
            report["results"].append({
                "saree_id": saree_id,
                "status": "failed",
                "reason": "Missing image filename"
            })
            continue

        # Prevent paths in the catalog from escaping SOURCE_DIR
        safe_filename = Path(filename).name

        source_path = SOURCE_DIR / safe_filename
        output_path = OUTPUT_DIR / f"{saree_id}.jpg"

        if not source_path.is_file():
            report["missing"] += 1
            report["results"].append({
                "saree_id": saree_id,
                "image": safe_filename,
                "status": "missing"
            })
            continue

        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report["failed"] += 1
            report["results"].append({
                "saree_id": saree_id,
                "image": safe_filename,
                "status": "failed",
                "reason": "Unsupported image extension"
            })
            continue

        try:
            preprocess_image(source_path, output_path)

            report["processed"] += 1
            report["results"].append({
                "saree_id": saree_id,
                "source_image": safe_filename,
                "processed_image": output_path.name,
                "status": "success"
            })

            print(f"[OK] {saree_id}: {safe_filename}")

        except (
            OSError,
            ValueError,
            UnidentifiedImageError
        ) as error:
            report["failed"] += 1
            report["results"].append({
                "saree_id": saree_id,
                "image": safe_filename,
                "status": "failed",
                "reason": str(error)
            })

            print(f"[ERROR] {saree_id}: {error}")

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n--- Image preprocessing completed ---")
    print(f"Catalog records : {report['total_catalog_records']}")
    print(f"Processed       : {report['processed']}")
    print(f"Missing         : {report['missing']}")
    print(f"Failed          : {report['failed']}")
    print(f"Output folder   : {OUTPUT_DIR}")
    print(f"Report          : {REPORT_PATH}")


if __name__ == "__main__":
    process_catalog()
