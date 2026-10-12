
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load local .env when present. Render environment variables
# remain available and are used when deploying.
load_dotenv(PROJECT_ROOT / ".env")


def csv_setting(name, default=""):
    return [
        value.strip().lower()
        for value in os.getenv(name, default).split(",")
        if value.strip()
    ]


APP_ENV = os.getenv("APP_ENV", "development")

CATALOG_BASE_URL = os.getenv(
    "CATALOG_BASE_URL",
    "https://www.rimasilks.com"
).rstrip("/")

ALLOWED_PRODUCT_DOMAINS = csv_setting(
    "ALLOWED_PRODUCT_DOMAINS",
    "www.rimasilks.com,rimasilks.com"
)

MODEL_NAME = os.getenv("MODEL_NAME", "ViT-B-32")
PRETRAINED_MODEL = os.getenv(
    "PRETRAINED_MODEL",
    "laion2b_s34b_b79k"
)

DEVICE = os.getenv("DEVICE", "auto").lower()

DEFAULT_MAX_PRICE = float(
    os.getenv("DEFAULT_MAX_PRICE", "3000")
)
DEFAULT_TOP_K = int(
    os.getenv("DEFAULT_TOP_K", "5")
)
MAX_TOP_K = int(
    os.getenv("MAX_TOP_K", "20")
)
MAX_UPLOAD_MB = int(
    os.getenv("MAX_UPLOAD_MB", "10")
)

APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if DEVICE not in {"auto", "cpu", "cuda"}:
    raise ValueError("DEVICE must be auto, cpu, or cuda.")

if DEFAULT_MAX_PRICE < 0:
    raise ValueError("DEFAULT_MAX_PRICE cannot be negative.")

if MAX_TOP_K < 1:
    raise ValueError("MAX_TOP_K must be at least 1.")

if not 1 <= DEFAULT_TOP_K <= MAX_TOP_K:
    raise ValueError(
        "DEFAULT_TOP_K must be between 1 and MAX_TOP_K."
    )

if MAX_UPLOAD_MB < 1:
    raise ValueError("MAX_UPLOAD_MB must be at least 1.")
