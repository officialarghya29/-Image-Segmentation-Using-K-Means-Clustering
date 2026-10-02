"""
main.py — Core K-Means image segmentation logic.

Every actual "image segmentation" operation lives in this file:
    - loading/resizing the uploaded image
    - running K-Means clustering (k-means++ init + Lloyd's algorithm)
    - rebuilding the segmented image from cluster centers
    - drawing the grid overlay
    - computing quantisation error / cluster legend
    - computing the elbow-graph error curve across a range of k

app.py only wires these functions up to HTTP routes — it contains no
clustering logic itself.
"""
import io
import base64

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError


# ---------------------------------------------------------------------------
# IMAGE LOADING
# ---------------------------------------------------------------------------

def load_image_from_bytes(file_bytes: bytes, max_size: int = 220) -> np.ndarray:
    """Decode uploaded image bytes into a resized RGB uint8 array."""
    if not file_bytes:
        raise ValueError("The uploaded image is empty.")
    if not isinstance(max_size, int) or not 1 <= max_size <= 500:
        raise ValueError("max_size must be between 1 and 500 pixels.")

    try:
        with Image.open(io.BytesIO(file_bytes)) as source:
            width, height = source.size
            if width < 1 or height < 1 or width * height > 40_000_000:
                raise ValueError("The image dimensions are unsupported (maximum 40 megapixels).")
            img = source.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("The image is too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc

    largest = max(width, height)
    if largest > max_size:
        scale = max_size / largest
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

    return np.asarray(img, dtype=np.uint8)


def _assign_clusters(pixels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign pixels to nearest center without allocating an (n, k, 3) array."""
    distances = (
        np.sum(pixels * pixels, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2 * pixels @ centers.T
    )
    np.maximum(distances, 0, out=distances)
    labels = np.argmin(distances, axis=1)
    return labels, distances[np.arange(pixels.shape[0]), labels]


def _validate_kmeans_input(pixels: np.ndarray, k: int, iterations: int) -> None:
    if not isinstance(pixels, np.ndarray) or pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("pixels must be a two-dimensional RGB array.")
    if pixels.shape[0] == 0:
        raise ValueError("At least one pixel is required.")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= pixels.shape[0]:
        raise ValueError("k must be between 1 and the number of pixels.")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels must contain only finite values.")


# ---------------------------------------------------------------------------
# K-MEANS CLUSTERING
# ---------------------------------------------------------------------------

def _kmeans_plusplus_init(pixels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialisation: spreads initial centers apart instead of
    picking them fully at random, which converges faster and more reliably."""
    n = pixels.shape[0]
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = pixels[rng.integers(n)]

    closest_dist_sq = np.sum((pixels - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = closest_dist_sq.sum()
        probs = closest_dist_sq / total if total > 0 else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centers[c] = pixels[idx]
        new_dist_sq = np.sum((pixels - centers[c]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centers


def kmeans(pixels: np.ndarray, k: int, iterations: int = 25, seed: int = 42):
    """Run K-Means on an (n, 3) array of normalised (0-1) RGB pixels.

    Returns:
        labels: (n,) cluster id per pixel
        centers: (k, 3) cluster center colors, normalised 0-1
        inertia: total squared-distance quantisation error
    """
    _validate_kmeans_input(pixels, k, iterations)
    pixels = np.asarray(pixels, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centers = _kmeans_plusplus_init(pixels, k, rng)
    labels = np.zeros(pixels.shape[0], dtype=np.int64)

    for _ in range(iterations):
        labels, _ = _assign_clusters(pixels, centers)

        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_centers[c] = pixels[mask].mean(axis=0)

        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    labels, min_distances = _assign_clusters(pixels, centers)
    inertia = float(np.sum(min_distances))

    return labels, centers, inertia


def segment_image(image_rgb: np.ndarray, k: int, iterations: int = 25):
    """Run K-Means segmentation on a full RGB image.

    Returns:
        segmented_image: (H, W, 3) uint8 image, each pixel repainted with
                          its cluster's color
        centers_255:     (k, 3) uint8 cluster center colors
        inertia:         total quantisation error (normalised 0-1 scale)
        pixel_count:      number of pixels clustered
    """
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.size == 0:
        raise ValueError("image_rgb must be a non-empty RGB image.")
    height, width, _ = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3).astype(np.float64) / 255.0

    labels, centers, inertia = kmeans(pixels, k, iterations)

    centers_255 = np.clip(centers * 255, 0, 255).astype(np.uint8)
    segmented_image = centers_255[labels].reshape(height, width, 3)

    return segmented_image, centers_255, inertia, pixels.shape[0]


# ---------------------------------------------------------------------------
# GRID OVERLAY
# ---------------------------------------------------------------------------

def add_grid(image_rgb: np.ndarray, spacing: int) -> np.ndarray:
    """Draw grid lines only (no text/values) over an image, spaced `spacing`
    pixels apart."""
    if not isinstance(spacing, (int, np.integer)) or spacing < 0:
        raise ValueError("spacing must be a non-negative integer.")
    if spacing == 0:
        return image_rgb

    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    for x in range(0, width + 1, spacing):
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255), width=1)
    for y in range(0, height + 1, spacing):
        draw.line([(0, y), (width, y)], fill=(255, 255, 255), width=1)

    return np.array(img)


# ---------------------------------------------------------------------------
# CLUSTER LEGEND AND ERROR STATS
# ---------------------------------------------------------------------------

def cluster_legend(centers_255: np.ndarray) -> list:
    """RGB + luminance (0.299R + 0.587G + 0.114B) for each cluster center."""
    legend = []
    for idx, (r, g, b) in enumerate(centers_255.tolist()):
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        legend.append({"cluster": idx, "r": r, "g": g, "b": b, "luminance": round(luminance, 1)})
    return legend


def error_stats(inertia: float, pixel_count: int) -> dict:
    """Quantisation error stats: mean error per pixel and RMSE on 0-255 scale."""
    if pixel_count <= 0:
        raise ValueError("pixel_count must be positive.")
    mean_error = inertia / pixel_count
    rmse_255 = float(np.sqrt(mean_error) * 255)
    return {
        "inertia": round(inertia, 6),
        "mean_error": round(mean_error, 8),
        "rmse_255": round(rmse_255, 2),
    }


# ---------------------------------------------------------------------------
# ELBOW GRAPH
# ---------------------------------------------------------------------------

def compute_elbow(image_rgb: np.ndarray, k_min: int, k_max: int,
                   sample_pixels: int = 3000, iterations: int = 15, seed: int = 42):
    """Run K-Means for a range of k values on a sampled subset of pixels,
    returning (k_values, mean_error_per_pixel) for plotting."""
    if k_min < 1 or k_max < k_min:
        raise ValueError("k_min must be positive and k_max must be at least k_min.")
    if sample_pixels < 1 or iterations < 1:
        raise ValueError("sample_pixels and iterations must be positive.")
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.size == 0:
        raise ValueError("image_rgb must be a non-empty RGB image.")
    all_pixels = image_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    n = all_pixels.shape[0]

    rng = np.random.default_rng(seed)
    if sample_pixels < n:
        idx = rng.choice(n, sample_pixels, replace=False)
        sample = all_pixels[idx]
    else:
        sample = all_pixels

    k_values, mean_errors = [], []
    for k in range(k_min, k_max + 1):
        if k > sample.shape[0]:
            break
        _, _, inertia = kmeans(sample, k, iterations, seed)
        k_values.append(k)
        mean_errors.append(inertia / sample.shape[0])

    return k_values, mean_errors


# ---------------------------------------------------------------------------
# ENCODING FOR HTTP RESPONSES
# ---------------------------------------------------------------------------

def image_to_base64_png(image_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array as a base64 PNG string for a JSON response."""
    img = Image.fromarray(image_rgb.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
