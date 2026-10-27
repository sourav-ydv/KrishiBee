import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
IMAGE_TYPES = (".jpg", ".jpeg", ".png")


def collect_pilot(root):
    items = []
    if not os.path.isdir(root):
        return items
    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(IMAGE_TYPES):
                items.append((os.path.join(folder, name), label))
    return items


def build_resnet(n_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


class Grab:
    def __init__(self):
        self.activation = None
        self.gradient = None

    def forward_hook(self, module, inputs, output):
        self.activation = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]


def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((384, 384))
    shown = image.resize((224, 224))
    tensor = EVAL_TF(image).unsqueeze(0)
    return shown, tensor


def predict(model, tensor):
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_index = int(probs.argmax().item())
    return pred_index, float(probs[pred_index].item())


def grad_cam(model, grab, tensor, class_index):
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    logits[0, class_index].backward()
    acts = grab.activation[0].detach()
    grads = grab.gradient[0].detach()
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam - cam.min()
    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    return cam.cpu().numpy()


def overlay(shown, cam):
    heat = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat = np.array(Image.fromarray(heat).resize(shown.size, Image.BILINEAR))
    color = plt.get_cmap("jet")(heat / 255.0)[..., :3]
    base = np.array(shown).astype(np.float32) / 255.0
    mixed = np.clip(0.45 * base + 0.55 * color, 0, 1)
    return (mixed * 255).astype(np.uint8)


def save_pair(shown, mixed, path, true_label, pred_label, confidence):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(shown)
    axes[0].set_title(f"Photo: {true_label}")
    axes[1].imshow(mixed)
    axes[1].set_title(f"Model looked here for {pred_label} ({confidence:.2f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/wheat_classifier_resnet18_seed0.pt")
    parser.add_argument("--pilot", default="eval_photos")
    parser.add_argument("--out", default="data/processed/plots/gradcam")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(args.checkpoint, map_location=device)
    classes = blob["classes"]
    if blob.get("arch", "resnet18") != "resnet18":
        raise SystemExit("This script is for a ResNet18 checkpoint.")

    model = build_resnet(len(classes)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    grab = Grab()
    layer = model.layer4[-1]
    layer.register_forward_hook(grab.forward_hook)
    layer.register_full_backward_hook(grab.backward_hook)

    items = collect_pilot(args.pilot)
    if not items:
        raise SystemExit(f"No photos found in {args.pilot}.")

    os.makedirs(args.out, exist_ok=True)
    saved_correct = set()
    rows = []
    for path, label in items:
        if label not in classes:
            print(f"Skipping {path}: label {label} is not in the checkpoint.")
            continue
        shown, tensor = prepare_image(path)
        tensor = tensor.to(device)
        with torch.no_grad():
            pred_index, confidence = predict(model, tensor)
        pred_label = classes[pred_index]
        correct = pred_label == label
        keep = (not correct) or (label not in saved_correct)
        if correct:
            saved_correct.add(label)
        if not keep:
            rows.append((os.path.basename(path), label, pred_label, confidence, correct, ""))
            continue
        kind = "wrong" if not correct else "correct"
        out_name = f"{kind}__{label}__pred-{pred_label}__{os.path.splitext(os.path.basename(path))[0]}.png"
        with torch.enable_grad():
            cam = grad_cam(model, grab, tensor, pred_index)
        mixed = overlay(shown, cam)
        save_pair(shown, mixed, os.path.join(args.out, out_name), label, pred_label, confidence)
        rows.append((os.path.basename(path), label, pred_label, confidence, correct, out_name))
        print(f"{kind:7}  true {label:16}  pred {pred_label:16}  {confidence:.2f}  {out_name}")

    index_path = os.path.join(args.out, "index.txt")
    wrong = [row for row in rows if not row[4]]
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Photos: {len(rows)}  Wrong: {len(wrong)}\n")
        f.write("These images are for slides. Do not commit them.\n\n")
        for name, label, pred, conf, correct, out_name in rows:
            mark = "ok" if correct else "WRONG"
            f.write(f"{mark:5}  true {label:16}  pred {pred:16}  {conf:.2f}  {out_name}\n")
    print(f"\nSaved slide images to {args.out}")
    print(f"Index: {index_path}")
    print(f"Wrong: {len(wrong)} of {len(rows)}")


if __name__ == "__main__":
    main()
