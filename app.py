"""
app.py — Flask web server.

This file only handles HTTP routing (serving the frontend, and exposing
/api/segment and /api/elbow). All of the actual image-segmentation work
happens in main.py.
"""
import os

from flask import Flask, request, jsonify, send_from_directory

import main as segmentation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/segment", methods=["POST"])
def api_segment():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    k = int(request.form.get("k", 4))
    max_size = int(request.form.get("max_size", 220))
    grid_size = int(request.form.get("grid_size", 30))

    if k < 2:
        return jsonify({"error": "k must be at least 2."}), 400

    image_rgb = segmentation.load_image_from_bytes(image_file.read(), max_size)
    segmented_image, centers_255, inertia, pixel_count = segmentation.segment_image(image_rgb, k)
    segmented_with_grid = segmentation.add_grid(segmented_image, grid_size)

    return jsonify({
        "width": int(image_rgb.shape[1]),
        "height": int(image_rgb.shape[0]),
        "original_png": segmentation.image_to_base64_png(image_rgb),
        "segmented_png": segmentation.image_to_base64_png(segmented_with_grid),
        "legend": segmentation.cluster_legend(centers_255),
        "error_stats": segmentation.error_stats(inertia, pixel_count),
    })


@app.route("/api/elbow", methods=["POST"])
def api_elbow():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    max_size = int(request.form.get("max_size", 220))
    k_min = int(request.form.get("k_min", 2))
    k_max = int(request.form.get("k_max", 9))

    if k_max < k_min:
        return jsonify({"error": "k_max must be >= k_min."}), 400

    image_rgb = segmentation.load_image_from_bytes(image_file.read(), max_size)
    k_values, mean_errors = segmentation.compute_elbow(image_rgb, k_min, k_max)

    return jsonify({"k_values": k_values, "mean_errors": mean_errors})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
