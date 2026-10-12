"""
Download saree products from Rima Silks and build the local catalog.

Run from the project root:
    python app/services/catalog_service.py

Output:
    saree_images/
    app/data/sarees.json
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_URL = "https://www.rimasilks.com"
ALLOWED_HOSTS = {"rimasilks.com", "www.rimasilks.com"}

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "app" / "data"
IMAGE_DIR = BASE_DIR / "saree_images"
OUTPUT_JSON = DATA_DIR / "sarees.json"

REQUEST_TIMEOUT = 25
REQUEST_DELAY_SECONDS = 0.5
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def create_directories():
    """Create output directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_page(url, timeout=REQUEST_TIMEOUT):
    """Fetch a page and raise a useful error for HTTP/network failures."""
    response = SESSION.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def normalize_product_url(url):
    """Normalize an allowed Rima Silks product URL."""
    absolute = urljoin(BASE_URL, url.strip())
    parsed = urlparse(absolute)

    if parsed.hostname not in ALLOWED_HOSTS:
        return None
    if not parsed.path.startswith("/products/"):
        return None

    # Canonical URL without query string or fragment.
    path = parsed.path.rstrip("/")
    return urlunparse(("https", "www.rimasilks.com", path, "", "", ""))


def get_all_product_urls():
    """
    Discover product URLs from Shopify's sitemap index.

    Shopify commonly uses sitemap_products_1.xml, but this function
    discovers sitemap names rather than relying on a fixed filename.
    """
    sitemap_url = f"{BASE_URL}/sitemap.xml"
    print(f"Reading sitemap index: {sitemap_url}")

    try:
        xml_text = get_page(sitemap_url)
        root = ET.fromstring(xml_text)
    except Exception as error:
        raise RuntimeError(
            f"Could not load or parse {sitemap_url}: {error}"
        ) from error

    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemap_urls = []

    if root.tag.endswith("sitemapindex"):
        for element in root.findall("sm:sitemap/sm:loc", namespace):
            if element.text:
                sitemap_urls.append(element.text.strip())
    elif root.tag.endswith("urlset"):
        # Some sites provide URLs directly in the root sitemap.
        sitemap_urls = [sitemap_url]
    else:
        raise RuntimeError(
            f"Unexpected sitemap XML root element: {root.tag}"
        )

    product_sitemaps = [
        url for url in sitemap_urls
        if "product" in url.lower()
    ]

    # If Shopify uses an unexpected sitemap filename, inspect the
    # sitemap URLs too rather than returning an empty catalog silently.
    if not product_sitemaps and root.tag.endswith("urlset"):
        product_sitemaps = [sitemap_url]

    if not product_sitemaps:
        raise RuntimeError(
            "No product sitemap was found in the sitemap index. "
            "Check the website's sitemap structure before continuing."
        )

    product_urls = set()

    for sitemap in product_sitemaps:
        print(f"Reading product sitemap: {sitemap}")
        try:
            sitemap_xml = get_page(sitemap)
            sitemap_root = ET.fromstring(sitemap_xml)

            # Support a sitemap that itself indexes more sitemaps.
            if sitemap_root.tag.endswith("sitemapindex"):
                nested_sitemaps = [
                    item.text.strip()
                    for item in sitemap_root.findall(
                        "sm:sitemap/sm:loc", namespace
                    )
                    if item.text
                ]
                for nested in nested_sitemaps:
                    try:
                        nested_xml = get_page(nested)
                        nested_root = ET.fromstring(nested_xml)
                        for entry in nested_root.findall("sm:url", namespace):
                            loc = entry.find("sm:loc", namespace)
                            if loc is not None and loc.text:
                                normalized = normalize_product_url(loc.text)
                                if normalized:
                                    product_urls.add(normalized)
                    except Exception as nested_error:
                        print(f"[WARN] Could not read {nested}: {nested_error}")
                    time.sleep(REQUEST_DELAY_SECONDS)
            else:
                for entry in sitemap_root.findall("sm:url", namespace):
                    loc = entry.find("sm:loc", namespace)
                    if loc is None or not loc.text:
                        continue
                    normalized = normalize_product_url(loc.text)
                    if normalized:
                        product_urls.add(normalized)

        except Exception as error:
            print(f"[WARN] Could not read sitemap {sitemap}: {error}")

        time.sleep(REQUEST_DELAY_SECONDS)

    urls = sorted(product_urls)
    print(f"Unique product URLs discovered: {len(urls)}")

    if not urls:
        raise RuntimeError(
            "The sitemap was read, but no /products/ URLs were found. "
            "Do not overwrite your existing catalog until sitemap "
            "discovery has been checked."
        )

    return urls


def _first_meta(soup, selectors):
    """Return the first non-empty metadata content matching selectors."""
    for attrs in selectors:
        tag = soup.find("meta", attrs=attrs)
        if tag:
            value = tag.get("content")
            if value and value.strip():
                return value.strip()
    return None


def _parse_price(value):
    """Parse a price only from a product-specific structured value."""
    if value is None:
        return None
    text = str(value).strip()
    # Structured Shopify prices are normally decimal values such as 2026.00.
    # Do not parse arbitrary page text.
    match = re.search(r"\d+(?:[.,]\d{1,2})?", text.replace(",", "."))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_product_data(product_url):
    """
    Extract product title, image and price from Shopify product page.

    Price stays None if no reliable structured product price is available.
    """
    html = get_page(product_url)
    soup = BeautifulSoup(html, "html.parser")

    name = _first_meta(soup, [
        {"property": "og:title"},
        {"name": "twitter:title"},
    ])

    if not name:
        title = soup.find("title")
        name = title.get_text(" ", strip=True) if title else None

    if name:
        name = re.sub(r"\s*[|–-]\s*Rima Silks\s*$", "", name, flags=re.I).strip()

    image_url = _first_meta(soup, [
        {"property": "og:image"},
        {"name": "twitter:image"},
    ])

    if not image_url:
        # Try Shopify product JSON embedded in the page.
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or script.get_text())
            except (json.JSONDecodeError, TypeError):
                continue

            records = data if isinstance(data, list) else [data]
            for record in records:
                if not isinstance(record, dict):
                    continue
                if record.get("@type") == "Product" or (
                    isinstance(record.get("@type"), list)
                    and "Product" in record.get("@type", [])
                ):
                    name = name or record.get("name")
                    image = record.get("image")
                    if isinstance(image, list):
                        image = image[0] if image else None
                    if isinstance(image, dict):
                        image = image.get("url")
                    image_url = image_url or image

    if image_url:
        image_url = urljoin(product_url, image_url)

    price_value = _first_meta(soup, [
        {"property": "product:price:amount"},
        {"itemprop": "price"},
    ])

    # Look for a JSON-LD Product offer price only when metadata is absent.
    if price_value is None:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or script.get_text())
            except (json.JSONDecodeError, TypeError):
                continue

            records = data if isinstance(data, list) else [data]
            for record in records:
                if not isinstance(record, dict):
                    continue
                offers = record.get("offers")
                if isinstance(offers, dict):
                    price_value = offers.get("price")
                    if price_value is None and isinstance(offers.get("lowPrice"), (str, int, float)):
                        price_value = offers.get("lowPrice")
                elif isinstance(offers, list):
                    for offer in offers:
                        if isinstance(offer, dict) and offer.get("price") is not None:
                            price_value = offer.get("price")
                            break
                if price_value is not None:
                    break
            if price_value is not None:
                break

    return {
        "name": name,
        "image_url": image_url,
        "price": _parse_price(price_value),
    }


def download_image(image_url, destination):
    """Download an image to destination. Return True only on success."""
    try:
        response = SESSION.get(
            image_url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            print(f"[WARN] URL did not return an image ({content_type}): {image_url}")
            response.close()
            return False

        temporary_path = destination.with_suffix(destination.suffix + ".part")
        total_bytes = 0
        max_bytes = 25 * 1024 * 1024  # 25 MB safety limit

        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError("Image exceeds 25 MB limit")
                file.write(chunk)

        response.close()

        if total_bytes == 0:
            temporary_path.unlink(missing_ok=True)
            return False

        # Validate that the downloaded file is an actual readable image.
        try:
            from PIL import Image
            with Image.open(temporary_path) as image:
                image.verify()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            print(f"[WARN] Downloaded file is not a valid image: {image_url}")
            return False

        temporary_path.replace(destination)
        return True

    except Exception as error:
        print(f"[WARN] Image download failed: {image_url} ({error})")
        try:
            temporary_path = destination.with_suffix(destination.suffix + ".part")
            temporary_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def scrape_catalog():
    create_directories()
    product_urls = get_all_product_urls()
    catalog = []

    total = len(product_urls)
    for index, product_url in enumerate(product_urls, start=1):
        try:
            product = extract_product_data(product_url)

            if not product.get("name") or not product.get("image_url"):
                print(f"[{index}/{total}] [SKIP] Missing name or image: {product_url}")
                continue

            # IDs are assigned from the source URL order. If a product fails,
            # gaps are harmless; each ID still remains stable for this run.
            image_filename = f"saree{index:05d}.jpg"
            image_path = IMAGE_DIR / image_filename

            if not download_image(product["image_url"], image_path):
                print(f"[{index}/{total}] [SKIP] Image download failed: {product_url}")
                continue

            catalog.append({
                "saree_id": f"S{index:05d}",
                "name": product["name"],
                "price": product["price"],
                "image": image_filename,
                "image_url": product["image_url"],
                "product_url": product_url,
            })

            price_display = (
                f"{product['price']:.2f}" if product["price"] is not None
                else "not found"
            )
            print(
                f"[{index}/{total}] Saved {image_filename} | "
                f"{product['name']} | price: {price_display}"
            )

        except Exception as error:
            print(f"[{index}/{total}] [ERROR] {product_url}: {error}")

        time.sleep(REQUEST_DELAY_SECONDS)

    if not catalog:
        raise RuntimeError(
            "No products were downloaded successfully. Existing catalog was not overwritten."
        )

    # Write to a temporary file first so an interrupted run does not corrupt JSON.
    temporary_json = OUTPUT_JSON.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=2, ensure_ascii=False)
    temporary_json.replace(OUTPUT_JSON)

    prices_found = sum(item["price"] is not None for item in catalog)

    print("\n--- Catalog download finished ---")
    print(f"Product URLs discovered: {total}")
    print(f"Products downloaded: {len(catalog)}")
    print(f"Products with structured prices: {prices_found}")
    print(f"Products without a reliable price: {len(catalog) - prices_found}")
    print(f"Catalog saved: {OUTPUT_JSON}")
    print(f"Images saved in: {IMAGE_DIR}")
    print(
        "\nNext: rerun your image processor and embedding generator, "
        "then verify the embedding IDs match the new catalog."
    )


if __name__ == "__main__":
    scrape_catalog()
