# Cluster Lens — K-Means Image Segmentation

A small web app that segments any image into k colour clusters using  
K-Means. Unlike the earlier version, **all the actual segmentation math now**  
**runs in Python on the backend**, not in the browser.

## What's in this folder

*   `main.py` — **all the image-segmentation logic**: loading/resizing the  
    image, K-Means clustering (k-means++ init + Lloyd's algorithm, in NumPy),  
    drawing the grid overlay, computing quantisation error and the elbow  
    curve, and encoding images for the API response.
*   `app.py` — the Flask web server. It only defines HTTP routes  
    (`/`, `/api/segment`, `/api/elbow`) and calls into `main.py` — it contains  
    no clustering logic itself.
*   `index.html` — the frontend. It uploads the chosen image to the backend  
    via `fetch()`/`FormData` and just displays whatever `main.py` sends back  
    (segmented image, cluster legend, error stats, elbow chart data). It also  
    includes a large responsive elbow graph and an **Error Analysis** tab that  
    lists every calculated mean quantisation-error value by k.
*   `requirements.txt` — Python dependencies: Flask, NumPy, Pillow.

## How a request flows

1.  You drop an image and click **Run segmentation** in the browser.
2.  The image + your settings (k, max size, grid spacing) are POSTed to  
    `/api/segment`.
3.  `app.py` reads the upload and calls functions in `main.py`  
    (`load_image_from_bytes` → `segment_image` → `add_grid` →  
    `cluster_legend` / `error_stats`).
4.  The backend returns JSON containing the original and segmented images  
    (base64 PNG), the cluster legend, and the error stats.
5.  The frontend renders that JSON — no clustering happens in JavaScript.

The **Graphs** tab works the same way against `/api/elbow`, which re-runs  
`main.py`'s `compute_elbow()` for a range of k values. The **Error Analysis**  
tab uses the exact k/error arrays returned by that same request and presents  
them in a table.

## Running it

```
# 1. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the backend
python app.py
```

Then open **http://localhost:5000** in your browser.

Change the port with:

```
PORT=8080 python app.py
```

To stop it, go back to the terminal and press `Ctrl+C`.

## Notes

*   Since segmentation now happens server-side, each "Run segmentation" or  
    "Plot error curve" click makes a real network request and can take a  
    moment on larger images or higher k — that's expected.
*   `debug=True` is set in `app.py` for local development; turn it off (or  
    use a production server like gunicorn) before deploying publicly.