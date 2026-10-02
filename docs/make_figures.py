"""
make_figures.py — regenerate every figure and metric used in README.md.

Run from the project root:

    python3 docs/make_figures.py

It builds a deterministic demo photograph, runs the real segmentation
code from main.py on it, and writes all PNGs into docs/. The measured
numbers are printed to stdout and saved to docs/metrics.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as segmentation  # noqa: E402

DOCS = os.path.dirname(os.path.abspath(__file__))


def build_demo_photo(width=640, height=420, seed=11):
    """Compose a deterministic sunset-over-mountains scene with gradients,
    soft haze and film grain so segmentation results look photographic."""
    rng = np.random.default_rng(seed)

    # Sky gradient: deep indigo -> warm orange near the horizon.
    top = np.array([24, 26, 74], dtype=np.float64)
    bottom = np.array([255, 168, 92], dtype=np.float64)
    rows = np.linspace(0, 1, height)[:, None, None]
    sky = top + (bottom - top) * rows ** 1.4
    canvas = np.repeat(sky, width, axis=1)

    # Water below the horizon with a cooler gradient.
    horizon = int(height * 0.62)
    w_top = np.array([92, 74, 110], dtype=np.float64)
    w_bottom = np.array([18, 20, 48], dtype=np.float64)
    wrows = np.linspace(0, 1, height - horizon)[:, None, None]
    canvas[horizon:] = w_top + (w_bottom - w_top) * wrows

    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Sun disc with a glow.
    sun = (int(width * 0.30), int(height * 0.34))
    radius = 46
    for r in range(radius + 60, radius, -3):
        alpha = int(70 * (1 - (r - radius) / 60))
        draw.ellipse([sun[0] - r, sun[1] - r, sun[0] + r, sun[1] + r],
                     fill=(min(255, 255 - alpha // 4), 190 - alpha // 3, 120 - alpha // 3))
    draw.ellipse([sun[0] - radius, sun[1] - radius, sun[0] + radius, sun[1] + radius],
                 fill=(255, 236, 190))

    # Two mountain layers (far then near).
    def ridge(base_y, peaks, colour):
        points = [(0, height)]
        for x, y in peaks:
            points.append((x, y))
        points.append((width, height))
        draw.polygon(points, fill=colour)

    ridge(horizon, [(0, horizon - 30), (110, horizon - 92), (210, horizon - 40),
                    (330, horizon - 118), (450, horizon - 55), (560, horizon - 88),
                    (width, horizon - 34)], (74, 62, 108))
    ridge(horizon, [(0, horizon - 6), (140, horizon - 52), (260, horizon - 14),
                    (400, horizon - 66), (520, horizon - 22), (width, horizon - 2)],
          (38, 30, 66))

    # Sun reflection streak on the water.
    for i in range(28):
        y = horizon + 6 + i * 7
        half = int(10 + i * 4)
        shade = int(200 - i * 5)
        draw.line([(sun[0] - half, y), (sun[0] + half, y)],
                  fill=(max(60, shade), max(40, shade - 30), max(30, shade - 60)), width=2)

    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float64)
    arr += rng.normal(0, 4.5, arr.shape)  # film grain
    return np.clip(arr, 0, 255).astype(np.uint8)


def save(arr, name, scale=2):
    img = Image.fromarray(arr.astype(np.uint8))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    img.save(os.path.join(DOCS, name))


def main():
    photo = build_demo_photo()
    small = segmentation.load_image_from_bytes(
        _to_png_bytes(photo), max_size=220)
    save(small, "00-original.png", scale=1)
    Image.fromarray(photo).save(os.path.join(DOCS, "00-source.png"))

    # k sweep
    tiles = [small]
    labels = ["input"]
    ks = [2, 4, 6, 8, 12, 16]
    for k in ks:
        seg, centers, inertia, n = segmentation.segment_image(small, k=k, iterations=30)
        save(seg, f"seg-k{k:02d}.png", scale=1)
        tiles.append(seg)
        labels.append(f"k={k}")

    # Contact sheet: input + each k, with white gutters.
    _contact_sheet(tiles, labels, os.path.join(DOCS, "01-k-sweep.png"))

    # Elbow curve + metrics.
    k_values, mean_errors = segmentation.compute_elbow(small, 2, 16, sample_pixels=40000, iterations=30)
    _elbow_chart(k_values, mean_errors, os.path.join(DOCS, "02-elbow-curve.png"))

    # Metrics table data for the fixed k values used in README.
    metrics = []
    n_pixels = small.shape[0] * small.shape[1]
    for k in ks:
        seg, centers, inertia, n = segmentation.segment_image(small, k=k, iterations=30)
        stats = segmentation.error_stats(inertia, n)
        metrics.append({
            "k": k,
            "inertia": stats["inertia"],
            "mean_error": stats["mean_error"],
            "rmse_255": stats["rmse_255"],
            "compression": round(n_pixels / k, 1),
        })

    # Cluster palette strips for k=8 and k=12.
    for k in (8, 12):
        _, centers, _, _ = segmentation.segment_image(small, k=k, iterations=30)
        _palette(centers, os.path.join(DOCS, f"03-palette-k{k}.png"), k)

    # Pipeline diagram.
    _pipeline(os.path.join(DOCS, "04-pipeline.png"))

    payload = {
        "image_width": int(small.shape[1]),
        "image_height": int(small.shape[0]),
        "pixel_count": int(n_pixels),
        "iterations": 30,
        "k_values": k_values,
        "mean_errors": mean_errors,
        "metrics": metrics,
    }
    with open(os.path.join(DOCS, "metrics.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


def _to_png_bytes(arr):
    import io
    buffer = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _contact_sheet(tiles, labels, path, cols=4):
    pad, label_h = 10, 26
    cell_w = max(t.shape[1] for t in tiles)
    cell_h = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * (cell_h + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (15, 18, 32))
    draw = ImageDraw.Draw(sheet)

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        col, row = index % cols, index // cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + label_h + pad)
        sheet.paste(Image.fromarray(tile.astype(np.uint8)), (x, y))
        draw.text((x + 2, y + cell_h + 6), label, fill=(190, 205, 240))
    sheet.save(path)


def _palette(centers_255, path, k):
    strip_w, strip_h = 64, 90
    sheet = Image.new("RGB", (strip_w * k + 20, strip_h + 46), (15, 18, 32))
    draw = ImageDraw.Draw(sheet)
    for i, (r, g, b) in enumerate(centers_255.tolist()):
        x = 10 + i * strip_w
        draw.rectangle([x, 34, x + strip_w - 8, 34 + strip_h], fill=(r, g, b))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        text = "#%02X%02X%02X" % (r, g, b)
        draw.text((x, 12), f"C{i}", fill=(190, 205, 240))
        draw.text((x, 34 + strip_h + 4), text, fill=(150, 165, 200))
        draw.text((x, 34 + strip_h + 18), f"L={lum:.0f}", fill=(120, 135, 175))
    sheet.save(path)


def _elbow_chart(k_values, mean_errors, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    fig.patch.set_facecolor("#0f1220")
    ax.set_facecolor("#0f1220")

    scaled = [e * 1e3 for e in mean_errors]
    ax.plot(k_values, scaled, "-o", color="#7c5cff", linewidth=2.4,
            markersize=6, markerfacecolor="#ffd28a", markeredgecolor="#7c5cff")
    ax.axvline(4, color="#ff7ab6", linestyle="--", linewidth=1.4)
    ax.annotate("elbow ≈ k=4", xy=(4, scaled[k_values.index(4)]),
                xytext=(6.2, max(scaled) * 0.72), color="#ff7ab6",
                arrowprops=dict(arrowstyle="->", color="#ff7ab6"))

    ax.set_xlabel("number of clusters  k", color="#c8d2f0")
    ax.set_ylabel("mean quantisation error  (×10⁻³)", color="#c8d2f0")
    ax.set_title("Elbow curve — error vs k", color="#ffffff", fontweight="bold")
    ax.grid(True, color="#2a3050", linewidth=0.7)
    ax.tick_params(colors="#9aa6d0")
    for spine in ax.spines.values():
        spine.set_color("#2a3050")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _pipeline(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = ["Upload\nimage", "Resize\n≤220 px", "Flatten to\nN×3 pixels",
             "K-Means++\ninit", "Lloyd\niterations", "Repaint\nclusters",
             "Grid +\nstats"]
    fig, ax = plt.subplots(figsize=(10.2, 2.1), dpi=140)
    fig.patch.set_facecolor("#0f1220")
    ax.set_facecolor("#0f1220")
    ax.axis("off")

    for i, step in enumerate(steps):
        x = i * 1.45
        ax.add_patch(plt.Rectangle((x, 0.1), 1.18, 0.8, facecolor="#1b2140",
                                   edgecolor="#7c5cff", linewidth=1.8,
                                   joinstyle="round", zorder=2))
        ax.text(x + 0.59, 0.5, step, ha="center", va="center",
                color="#e6ebff", fontsize=8.5, zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.4, 0.5), xytext=(x + 1.2, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#ffd28a", lw=2))
    ax.set_xlim(-0.2, len(steps) * 1.45)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
