import os
import random
import shutil
import sys

SOURCE = sys.argv[1] if len(sys.argv) > 1 else None
OUTPUT = "eval_photos"
PER_CLASS = 10
EXTENSIONS = (".jpg", ".jpeg", ".png")

KEYWORDS = {
    "yellow-rust": ["yellow", "stripe"],
    "brown-rust": ["brown", "leaf_rust", "leaf rust", "leafrust"],
    "powdery-mildew": ["mildew"],
    "healthy": ["healthy"],
}


def classify(path):
    lowered = path.lower()
    if "dwarf" in lowered:
        return None
    for label, words in KEYWORDS.items():
        if any(w in lowered for w in words):
            return label
    return None


def main():
    if not SOURCE or not os.path.isdir(SOURCE):
        print("Usage: python scripts/sample_eval.py <unzipped dataset folder>")
        return

    found = {label: [] for label in KEYWORDS}
    for root, _, files in os.walk(SOURCE):
        rel = os.path.relpath(root, SOURCE)
        for name in files:
            if not name.lower().endswith(EXTENSIONS):
                continue
            label = classify(name) or classify(rel)
            if label is not None:
                found[label].append(os.path.join(root, name))

    print({k: len(v) for k, v in found.items()})
    if any(len(v) == 0 for v in found.values()):
        print("A class has no images. Check the folder names and edit KEYWORDS.")
        return

    random.seed(42)
    for label, paths in found.items():
        chosen = random.sample(sorted(paths), min(PER_CLASS, len(paths)))
        out_dir = os.path.join(OUTPUT, label)
        os.makedirs(out_dir, exist_ok=True)
        for i, src in enumerate(chosen):
            ext = os.path.splitext(src)[1].lower()
            shutil.copy(src, os.path.join(out_dir, f"{label}_{i:02d}{ext}"))
    print(f"Copied up to {PER_CLASS} images per class into {OUTPUT}/")


if __name__ == "__main__":
    main()