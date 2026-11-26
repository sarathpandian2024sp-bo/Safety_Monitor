"""Fire and smoke models from D-Fire (Gaia) and the local image set.

D-Fire class order is smoke=0, fire=1. The local set uses that same order,
so a box can be merged no matter which model produced it.
"""

import os

DFIRE_REPO = "rabahdev/fire-smoke-yolov8n"
DFIRE_FILE = "best.pt"
LABEL_MAP = {
    "fire": "fire",
    "flame": "fire",
    "flames": "fire",
    "smoke": "smoke",
}


def local_weight_paths(base_dir):
    candidates = [
        os.path.join(base_dir, "custom_dataset", "weights", "best.pt"),
    ]
    runs = os.path.join(base_dir, "runs", "detect")
    found = []
    if os.path.isdir(runs):
        for root, _dirs, files in os.walk(runs):
            if os.path.basename(root) == "weights" and "best.pt" in files:
                found.append(os.path.join(root, "best.pt"))
        found.sort(key=os.path.getmtime)
    return candidates + found


def load_fire_models(download, yolo_cls, base_dir):
    """Load the D-Fire model, then the newest local weights trained from it."""
    if download is None or yolo_cls is None:
        raise RuntimeError("Install ultralytics and huggingface_hub to enable fire and smoke detection.")

    print("Loading D-Fire (Gaia) fire and smoke model...")
    weights = download(repo_id=DFIRE_REPO, filename=DFIRE_FILE)
    loaded = [{
        "name": "D-Fire",
        "detail": "Gaia, solutions on demand",
        "model": yolo_cls(weights),
        "path": weights,
    }]
    seen = {os.path.normcase(os.path.abspath(weights))}

    for path in local_weight_paths(base_dir):
        if not os.path.isfile(path):
            continue
        full = os.path.normcase(os.path.abspath(path))
        if full in seen:
            continue
        print(f"Loading local fire/smoke weights: {path}")
        loaded.append({
            "name": "custom",
            "detail": "local fire and smoke set",
            "model": yolo_cls(path),
            "path": path,
        })
        break
    return loaded


def model_message(models):
    names = [item["name"] for item in models]
    if "D-Fire" in names and "custom" in names:
        return "D-Fire (Gaia) and the local fire/smoke set are both on."
    if "D-Fire" in names:
        return "D-Fire (Gaia) fire and smoke detection is on."
    if names:
        return "Local fire and smoke detection is on."
    return "Fire and smoke model failed to load."


def canonical_label(name):
    return LABEL_MAP.get(str(name).lower().strip())


def _iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if overlap <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def merge_boxes(boxes, limit=0.45):
    """Keep the stronger box when both datasets mark the same fire or smoke."""
    ordered = sorted(boxes, key=lambda box: box["conf"], reverse=True)
    kept = []
    for box in ordered:
        if any(
            box["label"] == old["label"] and _iou(box["xyxy"], old["xyxy"]) >= limit
            for old in kept
        ):
            continue
        kept.append(box)
    return kept
