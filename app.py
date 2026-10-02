"""Flask routes for the Cluster Lens image-segmentation app."""
import os

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

import main as segmentation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify({"error": "Image exceeds the 10 MB upload limit."}), 413


def _form_int(name, default, minimum, maximum):
    """Read and validate an integer form value, raising a useful error."""
    raw_value = request.form.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _load_uploaded_image(max_size):
    if "image" not in request.files:
        raise ValueError("No image file provided.")
    image_file = request.files["image"]
    if not image_file.filename:
        raise ValueError("Choose an image file before submitting.")
    return segmentation.load_image_from_bytes(image_file.read(), max_size)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/segment", methods=["POST"])
def api_segment():
    try:
        k = _form_int("k", 4, 2, 16)
        max_size = _form_int("max_size", 220, 40, 500)
        grid_size = _form_int("grid_size", 30, 0, 200)
        image_rgb = _load_uploaded_image(max_size)
        segmented_image, centers_255, inertia, pixel_count = segmentation.segment_image(image_rgb, k)
        segmented_with_grid = segmentation.add_grid(segmented_image, grid_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

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
    try:
        max_size = _form_int("max_size", 220, 40, 500)
        k_min = _form_int("k_min", 2, 2, 16)
        k_max = _form_int("k_max", 9, 2, 16)
        if k_max < k_min:
            raise ValueError("k_max must be greater than or equal to k_min.")
        image_rgb = _load_uploaded_image(max_size)
        k_values, mean_errors = segmentation.compute_elbow(image_rgb, k_min, k_max)
        if not k_values:
            raise ValueError("The image has too few pixels for the selected cluster range.")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"k_values": k_values, "mean_errors": mean_errors})


if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", "5000"))
    except ValueError as exc:
        raise SystemExit("PORT must be a whole number between 1 and 65535.") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be a whole number between 1 and 65535.")
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=debug)
