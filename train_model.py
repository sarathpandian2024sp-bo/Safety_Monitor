"""Fine-tune the D-Fire (Gaia) weights on the local fire/smoke images.

Both sets use smoke=0 and fire=1. Training starts from the D-Fire checkpoint,
so the local images are added on top of that dataset instead of replacing it.
"""

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from fire_datasets import DFIRE_FILE, DFIRE_REPO

weights = hf_hub_download(repo_id=DFIRE_REPO, filename=DFIRE_FILE)
model = YOLO(weights)
model.train(
    data="custom_dataset/data.yaml",
    epochs=40,
    imgsz=416,
    batch=8,
    device="cpu",
)
