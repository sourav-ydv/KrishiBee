import base64
import csv
import io
import math
import os
import time

import requests
from PIL import Image

API_URL = os.environ.get("DIAGNOSE_URL", "http://localhost:8000/diagnose")
PHOTO_DIR = "eval_photos"
OUTPUT_CSV = "data/processed/diagnose_eval.csv"
PAUSE_SECONDS = 25
MAX_SIDE = 1024
DISEASES = ["yellow-rust", "brown-rust", "powdery-mildew"]
IMAGE_TYPES = (".jpg", ".jpeg", ".png")


def encode(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def call(image_b64):
    for _ in range(4):
        resp = requests.post(API_URL, json={"crop": "Wheat", "image_base64": image_b64}, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 502):
            time.sleep(40)
            continue
        print("  Request failed:", resp.status_code, resp.text[:200])
        return None
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt(k, n):
    low, high = wilson(k, n)
    return f"{k}/{n} (95% CI {100 * low:.0f}-{100 * high:.0f}%)"


def main():
    if not os.path.isdir(PHOTO_DIR):
        print("No eval_photos folder. Run scripts/sample_eval.py first.")
        return

    labels = sorted(d for d in os.listdir(PHOTO_DIR) if os.path.isdir(os.path.join(PHOTO_DIR, d)))
    files = [(l, n) for l in labels for n in sorted(os.listdir(os.path.join(PHOTO_DIR, l))) if n.lower().endswith(IMAGE_TYPES)]

    rows = []
    for i, (label, name) in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {label}/{name}")
        data = call(encode(os.path.join(PHOTO_DIR, label, name)))
        if data is None:
            rows.append({"label": label, "file": name, "status": "error", "predicted": "", "observations": ""})
        else:
            rows.append({
                "label": label,
                "file": name,
                "status": "ok" if data["photo_ok"] else "unreadable",
                "predicted": "|".join(c["id"] for c in data["candidates"]),
                "observations": data["observations"],
            })
        time.sleep(PAUSE_SECONDS)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "file", "status", "predicted", "observations"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== RESULTS ===")
    all_n = all_top1 = all_top3 = 0
    for label in labels:
        group = [r for r in rows if r["label"] == label and r["status"] != "error"]
        n = len(group)
        if label in DISEASES:
            top1 = sum(1 for r in group if r["predicted"].split("|")[0] == label)
            top3 = sum(1 for r in group if label in r["predicted"].split("|"))
            nomatch = sum(1 for r in group if r["status"] == "ok" and not r["predicted"])
            unreadable = sum(1 for r in group if r["status"] == "unreadable")
            print(f"{label}: top-1 {fmt(top1, n)}, top-3 {fmt(top3, n)}, no match {nomatch}, unreadable {unreadable}")
            all_n += n
            all_top1 += top1
            all_top3 += top3
        elif label == "healthy":
            clean = sum(1 for r in group if r["status"] == "ok" and not r["predicted"])
            print(f"healthy: no false alarm {fmt(clean, n)}")
    if all_n:
        print(f"All diseased photos: top-1 {fmt(all_top1, all_n)}, top-3 {fmt(all_top3, all_n)}")


if __name__ == "__main__":
    main()