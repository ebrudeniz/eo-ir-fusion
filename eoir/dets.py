"""
Tespit dosyasi (predict_raw bicimi) ve etiket okuma
===================================================

Baseline'lar ve fuzyon ayni JSON bicimini uretir, boylece hepsi ayni
degerlendiriciyle olculur:

{
  "meta":   {"modality": "ir", "split": "val", "conf": 0.001, ...},
  "images": {"050001": {"boxes": [[x1, y1, x2, y2], ...], "scores": [...]}, ...}
}

Kutular ORIJINAL goruntu pikselinde, xyxy. Tespiti olmayan bir goruntu
"images" icinde bos listeyle yer alir; hic yer almamasi "bu goruntu
islenmedi" demektir ve degerlendirici bunu hata sayar.
Dosya adi .gz ile bitiyorsa gzip ile yazilir/okunur.
"""

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLIP_DIGITS = 2  # "050001" -> klip "05" (prepare_llvip_yolo.py ile ayni varsayim)


def clip_of(stem: str) -> str:
    return stem[:CLIP_DIGITS]


def _open(path: Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def save_dets(path, images: dict, meta: dict):
    """images: {stem: (boxes[N,4], scores[N])}. Kutular 2, skorlar 6 haneye yuvarlanir."""
    out = {"meta": meta, "images": {}}
    for stem in sorted(images):
        boxes, scores = images[stem]
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        out["images"][stem] = {
            "boxes": np.round(boxes, 2).tolist(),
            "scores": np.round(scores, 6).tolist(),
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))


def load_dets(path):
    """(images, meta) dondurur. images: {stem: (boxes[N,4] float64, scores[N] float64)}."""
    with _open(path, "r") as f:
        raw = json.load(f)
    images = {}
    for stem, d in raw["images"].items():
        boxes = np.asarray(d["boxes"], dtype=float).reshape(-1, 4)
        scores = np.asarray(d["scores"], dtype=float).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError(f"{path}: {stem} icin kutu ve skor sayisi farkli")
        images[stem] = (boxes, scores)
    return images, raw.get("meta", {})


def load_gt(dataset_dir, split: str):
    """
    YOLO etiketlerini orijinal piksel xyxy'ye cevirir.
    {stem: boxes[M,4]} dondurur. Boyut goruntu dosyasinin basligindan okunur
    (PIL yalnizca basligi okur, goruntuyu acmaz).
    """
    img_dir = Path(dataset_dir) / "images" / split
    lbl_dir = Path(dataset_dir) / "labels" / split
    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"'{img_dir}' icinde .jpg yok")

    gt = {}
    for img_path in img_paths:
        with Image.open(img_path) as im:
            w, h = im.size
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            raise FileNotFoundError(f"Etiket yok: {lbl_path}")
        rows = [r.split() for r in lbl_path.read_text(encoding="utf-8").splitlines() if r.strip()]
        boxes = np.zeros((len(rows), 4))
        for k, r in enumerate(rows):
            xc, yc, bw, bh = (float(v) for v in r[1:5])
            boxes[k] = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
        gt[img_path.stem] = boxes
    return gt


def check_same_images(gt: dict, images: dict, name: str = "tespit"):
    """Tespit dosyasi etiketlerle tam ayni goruntu kumesini kapsamali."""
    missing = sorted(set(gt) - set(images))
    extra = sorted(set(images) - set(gt))
    if missing or extra:
        raise ValueError(
            f"{name}: goruntu kumesi etiketlerle uyusmuyor. "
            f"eksik={len(missing)} (orn. {missing[:3]}), fazla={len(extra)} (orn. {extra[:3]})"
        )
