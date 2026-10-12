# Saree Matcher

Saree Matcher is a FastAPI web application that recommends sarees based on an uploaded blouse-piece photo. It combines visual similarity from OpenCLIP with dominant-color similarity, filters products by budget, and displays matching products from the local catalog.

## Project structure

```text
.
├── app/
│   ├── data/
│   │   ├── sarees.json                 # Product metadata
│   │   ├── saree_embeddings.npy        # Precomputed catalog image vectors
│   │   └── saree_embedding_ids.json    # IDs in the same order as the vectors
│   ├── services/
│   │   ├── catalog_service.py          # Scrapes product data and source images
│   │   ├── image_processor.py          # Creates normalized catalog image files
│   │   ├── embedding_service.py        # Generates OpenCLIP catalog embeddings
│   │   ├── similarity_service.py       # Searches visual embeddings
│   │   ├── color_service.py            # Computes color-palette similarity
│   │   └── recommendation_service.py   # Combines scores and applies the budget
│   ├── templates/
│   │   └── index.html                  # Upload form and recommendations page
│   ├── config.py                       # Environment-based settings
│   └── main.py                         # FastAPI application and routes
├── saree_images/                       # Original catalog images
├── processed_images/                   # Resized images shown by the web app
├── uploads/                            # Temporary uploaded images (created as needed)
├── processing_report.json              # Latest image-processing report
└── requirements.txt                    # Python dependencies
```

The catalog JSON, embedding files, and image directories need to stay in sync. The app needs `processed_images/` for the product pictures and uses the original files in `saree_images/` to calculate color similarity.

## Requirements

- Python 3.10 or newer
- Internet access the first time the OpenCLIP pretrained model is loaded, so its weights can be downloaded
- A CUDA-capable PyTorch setup is optional; the app can run on CPU

## Install and run on Windows

Open PowerShell in the project root and create an environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the web server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

When the server is ready, open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. Stop the development server with **Ctrl+C**.

The first startup loads the OpenCLIP model (`ViT-B-32`, pretrained weights `laion2b_s34b_b79k`) and may take longer while the model is downloaded and cached. The model and catalog embeddings are loaded when the application starts.

## Use the application

1. Choose a blouse-piece image in JPG, PNG, or WEBP format. Images must be 10 MB or smaller.
2. Enter a maximum budget in rupees.
3. Choose how many recommendations to show (1–20).
4. Select **Find Matching Sarees**.
5. Compare the recommended sarees by name, price, visual similarity, and color similarity. Use **View product** to open an allowed product link when one is available.

Products without a known price are excluded when a budget is provided. If no products fit, increase the budget and try again. Uploaded images are written temporarily while matching and removed after processing.

## Configuration

The app reads optional overrides from a `.env` file in the project root. The defaults below apply when a setting is not provided.

| Setting | Default | Description |
| --- | --- | --- |
| `DEVICE` | `auto` | `auto` selects CUDA when available and CPU otherwise; can also be set to `cpu` or `cuda`. |
| `MODEL_NAME` | `ViT-B-32` | OpenCLIP image model used for query images. |
| `PRETRAINED_MODEL` | `laion2b_s34b_b79k` | OpenCLIP pretrained weights. |
| `DEFAULT_MAX_PRICE` | `3000` | Initial budget shown in the form. |
| `DEFAULT_TOP_K` | `5` | Initial number of recommendations shown. |
| `MAX_TOP_K` | `20` | Maximum number of recommendations allowed. |
| `MAX_UPLOAD_MB` | `10` | Maximum accepted upload size in megabytes. |
| `ALLOWED_PRODUCT_DOMAINS` | `www.rimasilks.com,rimasilks.com` | Comma-separated HTTPS domains whose product links can be displayed. |

For example, to force CPU use, create `.env` in the project root with:

```dotenv
DEVICE=cpu
```

Set `DEVICE=cuda` only when CUDA is available in the installed PyTorch environment. The app reports an error at startup if CUDA is explicitly requested but unavailable.

## Refreshing the catalog (optional)

The checked-in catalog assets are ready to run. To replace or refresh them, run these commands from the project root in order:

```powershell
.\.venv\Scripts\python.exe -m app.services.catalog_service
.\.venv\Scripts\python.exe -m app.services.image_processor
.\.venv\Scripts\python.exe -m app.services.embedding_service
```

The catalog scraper downloads product metadata and original images into `app/data/sarees.json` and `saree_images/`. Image processing creates `processed_images/` and updates `processing_report.json`; embedding generation recreates the `.npy` vectors and ID mapping. Ensure all catalog images process successfully before starting the app.
