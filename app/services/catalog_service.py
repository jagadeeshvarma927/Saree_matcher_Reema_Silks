
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.rimasilks.com"

OUTPUT_JSON = Path("app/data/sarees.json")
IMAGE_DIR = Path("saree_images")

# Start small for testing
MAX_PRODUCTS = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36"
    )
}


COLLECTIONS = [
    "/collections/chiffon-sarees",
    "/collections/georgette-sarees",
    "/collections/mashru-crepe-sarees",
    "/collections/mashru-silk-sarees",
    "/collections/tissue-silk-sarees",
    "/collections/office-wear-sarees",
    "/collections/kota-sarees",
]


def create_directories():
    """Create required folders."""

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_page(url):
    """Download a webpage."""

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return response.text


def extract_product_links(collection_url):
    """
    Extract product URLs from a Rima Silks collection page.
    """

    print(f"\nReading collection:")
    print(collection_url)

    html = get_page(collection_url)

    soup = BeautifulSoup(html, "lxml")

    product_links = set()

    for link in soup.find_all("a", href=True):

        href = link["href"]

        if "/products/" in href:

            product_url = urljoin(BASE_URL, href)

            product_links.add(product_url)

    print(f"Found {len(product_links)} products")

    return list(product_links)


def extract_price(text):
    """
    Extract numeric price from text.
    Example:
        Rs. 3,999.00 INR
        -> 3999.0
    """

    if not text:
        return None

    matches = re.findall(
        r"[\d,]+(?:\.\d+)?",
        text
    )

    if not matches:
        return None

    value = matches[-1]

    value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def extract_product_data(product_url):
    """
    Extract product information from a product page.
    """

    print(f"  Processing: {product_url}")

    html = get_page(product_url)

    soup = BeautifulSoup(html, "lxml")

    # -------------------------------------------------
    # Product title
    # -------------------------------------------------

    title = None

    og_title = soup.find(
        "meta",
        property="og:title"
    )

    if og_title:
        title = og_title.get("content")

    if not title:

        h1 = soup.find("h1")

        if h1:
            title = h1.get_text(
                strip=True
            )

    # -------------------------------------------------
    # Product image
    # -------------------------------------------------

    image_url = None

    og_image = soup.find(
        "meta",
        property="og:image"
    )

    if og_image:

        image_url = og_image.get("content")

    # -------------------------------------------------
    # Product price
    # -------------------------------------------------

    price = None

    price_meta = soup.find(
        "meta",
        property="product:price:amount"
    )

    if price_meta:

        try:
            price = float(
                price_meta.get("content")
            )
        except (ValueError, TypeError):
            price = None

    # Fallback price search
    if price is None:

        page_text = soup.get_text(
            " ",
            strip=True
        )

        price = extract_price(
            page_text
        )

    # -------------------------------------------------
    # Return data
    # -------------------------------------------------

    return {
        "name": title,
        "price": price,
        "product_url": product_url,
        "image_url": image_url
    }


def download_image(image_url, file_path):
    """
    Download product image.
    """

    if not image_url:

        print("    No image URL found")

        return False

    try:

        response = requests.get(
            image_url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        with open(file_path, "wb") as file:

            file.write(
                response.content
            )

        return True

    except Exception as error:

        print(
            f"    Image download failed: {error}"
        )

        return False


def scrape_catalog():

    create_directories()

    all_products = {}

    # -------------------------------------------------
    # Discover products
    # -------------------------------------------------

    for collection in COLLECTIONS:

        collection_url = urljoin(
            BASE_URL,
            collection
        )

        try:

            links = extract_product_links(
                collection_url
            )

            for link in links:

                if link not in all_products:

                    all_products[link] = True

                if len(all_products) >= MAX_PRODUCTS:

                    break

        except Exception as error:

            print(
                f"Collection failed: {error}"
            )

        if len(all_products) >= MAX_PRODUCTS:

            break

    print(
        f"\nTotal unique products found: "
        f"{len(all_products)}"
    )

    # -------------------------------------------------
    # Extract metadata
    # -------------------------------------------------

    catalog = []

    for index, product_url in enumerate(
        all_products.keys(),
        start=1
    ):

        if index > MAX_PRODUCTS:

            break

        try:

            product = extract_product_data(
                product_url
            )

            if not product["name"]:

                print(
                    "    Skipping product without name"
                )

                continue

            # -----------------------------------------
            # File name
            # -----------------------------------------

            image_filename = (
                f"saree{index:03d}.jpg"
            )

            image_path = (
                IMAGE_DIR /
                image_filename
            )

            # -----------------------------------------
            # Download image
            # -----------------------------------------

            success = download_image(
                product["image_url"],
                image_path
            )

            if not success:

                continue

            # -----------------------------------------
            # Catalog record
            # -----------------------------------------

            record = {

                "saree_id":
                    f"S{index:03d}",

                "name":
                    product["name"],

                "price":
                    product["price"],

                "image":
                    image_filename,

                "image_url":
                    product["image_url"],

                "product_url":
                    product["product_url"]
            }

            catalog.append(record)

            print(
                f"    Saved: {image_filename}"
            )

            # Be polite to the website
            time.sleep(1)

        except Exception as error:

            print(
                f"    Product failed: {error}"
            )

    # -------------------------------------------------
    # Save JSON
    # -------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            catalog,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n================================")
    print("Catalog creation completed")
    print("================================")

    print(
        f"Products saved: {len(catalog)}"
    )

    print(
        f"Metadata: {OUTPUT_JSON}"
    )

    print(
        f"Images: {IMAGE_DIR}"
    )


if __name__ == "__main__":

    scrape_catalog()