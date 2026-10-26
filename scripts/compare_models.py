import csv
import math
import os

VLM_CSV = "data/processed/diagnose_eval.csv"
ARCHS = ["resnet18", "mobilenet_v3_small"]
DISEASES = ["yellow-rust", "brown-rust", "powdery-mildew"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["label"], r["file"]): r for r in csv.DictReader(f)}


def mcnemar_exact(only_a, only_b):
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def vlm_first(row):
    return row["predicted"].split("|")[0] if row["predicted"] else "none"


def main():
    if not os.path.exists(VLM_CSV):
        print(f"Missing {VLM_CSV}. Run scripts/eval_diagnose.py first.")
        return
    vlm = read_rows(VLM_CSV)

    for arch in ARCHS:
        path = f"data/processed/classifier_pilot_{arch}.csv"
        if not os.path.exists(path):
            print(f"Missing {path}. Train the {arch} classifier first.")
            continue
        clf = read_rows(path)

        diseased = sorted(k for k in vlm if k in clf and k[0] in DISEASES)
        both = only_clf = only_vlm = neither = 0
        disagreements = []
        for key in diseased:
            label = key[0]
            v_pred = vlm_first(vlm[key])
            c_pred = clf[key]["predicted"]
            v_ok, c_ok = v_pred == label, c_pred == label
            if v_ok and c_ok:
                both += 1
            elif c_ok:
                only_clf += 1
            elif v_ok:
                only_vlm += 1
            else:
                neither += 1
            if v_ok != c_ok:
                disagreements.append((key[1], label, v_pred, c_pred, clf[key]["confidence"]))

        print(f"\n=== {arch} classifier vs vision model, {len(diseased)} diseased pilot photos ===")
        print(f"Both correct: {both}   only classifier: {only_clf}   only vision model: {only_vlm}   neither: {neither}")
        print(f"Classifier correct: {both + only_clf}/{len(diseased)}   vision model correct: {both + only_vlm}/{len(diseased)}")
        print(f"Exact McNemar test on the {only_clf + only_vlm} photos where they differ: p = {mcnemar_exact(only_clf, only_vlm):.3f}")
        if disagreements:
            print("Photos where exactly one was right (file, true, vision model, classifier, classifier confidence):")
            for row in disagreements:
                print("  ", *row)

        healthy = sorted(k for k in vlm if k in clf and k[0] == "healthy")
        v_healthy = sum(1 for k in healthy if vlm[k]["status"] == "ok" and not vlm[k]["predicted"])
        c_healthy = sum(1 for k in healthy if clf[k]["predicted"] == "healthy")
        print(f"Healthy photos handled correctly: vision model {v_healthy}/{len(healthy)}, classifier {c_healthy}/{len(healthy)}")


if __name__ == "__main__":
    main()