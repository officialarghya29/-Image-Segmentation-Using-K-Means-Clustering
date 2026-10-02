"""End-to-end API + asset integrity scan for Cluster Lens."""
import io
import json
import os
import re
import sys
import urllib.request

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:5000"
# project root is the parent of this script's docs/ directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def multipart(fields, files):
    boundary = "----clusterlensscan"
    body = b""
    for key, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    for key, (filename, data) in files.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\";"
                 f" filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def post(path, fields, files):
    body, ctype = multipart(fields, files)
    request = urllib.request.Request(BASE + path, data=body,
                                     headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def png(width, height, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.linspace(0, 1, height)[:, None, None]
    arr = (rows * np.array([220.0, 180.0, 90.0]) + rng.normal(0, 3, (height, width, 3)))
    buffer = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------- page load
with urllib.request.urlopen(BASE + "/", timeout=30) as response:
    html = response.read().decode()
check("GET / returns 200 and the app shell", response.status == 200 and "Cluster Lens" in html)

# ---------------------------------------------------------------- happy path
status, data = post("/api/segment", {"k": 4, "max_size": 220, "grid_size": 30},
                    {"image": ("test.png", png(300, 200))})
check("POST /api/segment returns 200", status == 200, f"status={status}")
check("segment response has all keys",
      set(data) >= {"width", "height", "original_png", "segmented_png", "legend", "error_stats"},
      str(sorted(data)))
check("segment legend length equals k", len(data.get("legend", [])) == 4)
check("segment error stats are sane",
      data["error_stats"]["inertia"] > 0 and data["error_stats"]["rmse_255"] > 0,
      json.dumps(data.get("error_stats")))
for key in ("original_png", "segmented_png"):
    raw = __import__("base64").b64decode(data[key])
    img = Image.open(io.BytesIO(raw))
    check(f"{key} decodes as PNG", img.format == "PNG", f"{img.size}")

# ---------------------------------------------------------------- elbow
status, elbow = post("/api/elbow", {"k_min": 2, "k_max": 6, "max_size": 160},
                     {"image": ("test.png", png(300, 200, seed=2))})
check("POST /api/elbow returns 200", status == 200, f"status={status}")
check("elbow returns matching arrays",
      elbow["k_values"] == [2, 3, 4, 5, 6] and len(elbow["mean_errors"]) == 5,
      str(elbow.get("k_values")))
check("elbow errors decrease", elbow["mean_errors"][0] > elbow["mean_errors"][-1])

# ---------------------------------------------------------------- bad input
bad_cases = [
    ("k out of range", {"k": 99, "max_size": 220, "grid_size": 0}, {"image": ("t.png", png(50, 50))}),
    ("k below minimum", {"k": 1, "max_size": 220, "grid_size": 0}, {"image": ("t.png", png(50, 50))}),
    ("k not a number", {"k": "abc", "max_size": 220, "grid_size": 0}, {"image": ("t.png", png(50, 50))}),
    ("max_size out of range", {"k": 4, "max_size": 5000, "grid_size": 0}, {"image": ("t.png", png(50, 50))}),
    ("grid_size negative", {"k": 4, "max_size": 220, "grid_size": -5}, {"image": ("t.png", png(50, 50))}),
    ("not an image", {"k": 4, "max_size": 220, "grid_size": 0}, {"image": ("t.png", b"garbage bytes")}),
    ("no file", {"k": 4, "max_size": 220, "grid_size": 0}, {}),
]
for name, fields, files in bad_cases:
    status, body = post("/api/segment", fields, files)
    check(f"rejects {name} with 400 JSON",
          status == 400 and "error" in body, f"status={status} body={body}")

status, body = post("/api/elbow", {"k_min": 8, "k_max": 3, "max_size": 160},
                    {"image": ("t.png", png(80, 80))})
check("elbow rejects k_max < k_min", status == 400 and "error" in body, f"status={status}")

# ---------------------------------------------------------------- README assets
readme = open(os.path.join(ROOT, "README.md")).read()
missing = [src for src in re.findall(r'src="([^"]+)"', readme)
           if not os.path.exists(os.path.join(ROOT, src))]
check("every README image exists on disk", not missing, f"missing: {missing}")

# ---------------------------------------------------------------- frontend sanity
check("index.html references both API endpoints",
      "/api/segment" in html and "/api/elbow" in html)
check("index.html lists all five roll numbers",
      all(roll in html for roll in
          ["24155040", "24155380", "24155758", "24155156", "24155983"]))
check("index.html has no unclosed fetch error handling",
      html.count("res.json().catch") >= 2)
check("index.html enforces the 10 MB limit in copy", "max 10 MB" in html)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")