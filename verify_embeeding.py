
import json
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent
data_dir = root / "app" / "data"

embeddings = np.load(
    data_dir / "saree_embeddings.npy"
)

with open(
    data_dir / "saree_embedding_ids.json",
    "r",
    encoding="utf-8"
) as file:
    saree_ids = json.load(file)

print("Embedding shape:", embeddings.shape)
print("Number of IDs:", len(saree_ids))
print("Data type:", embeddings.dtype)
print("All finite:", np.isfinite(embeddings).all())

print("\nFirst 5 saree IDs:")
print(saree_ids[:5])

assert embeddings.ndim == 2
assert embeddings.shape[0] == len(saree_ids)
assert np.isfinite(embeddings).all()

print("\nEmbedding verification successful!")