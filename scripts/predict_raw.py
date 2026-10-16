"""
Ham tespit kaydi
================

Bir dedektoru (best.pt) bir veri kumesi bolumunde calistirip TUM adaylari
(conf >= 0.001) eoir/dets.py bicimindeki JSON'a yazar. Metrik HESAPLAMAZ:
test bolumunde de guvenle calistirilabilir, test'e "bakmis" olmayiz.

Neden esikten gecmemis kutular da: fuzyon bir sensorun dusuk guvenle
gordugu ama digerinin yuksek guvenle onayladigi nesneyi kurtarabilir.

Ultralytics val ile ayni girdi: val, 1280x1024 goruntuyu 640x512'ye
kuculttukten sonra rect modunda 640x544'e doldurur (pad=0.5). Burada
predict'e imgsz=(544, 640) verince ayni yeniden boyutlama (INTER_LINEAR) ve
ayni ortalanmis dolgu uygulanir. Boylece saglama adiminda kalan fark
olcum kodundan gelir, on islemeden degil.

Kullanim (Colab, GPU):
  python3 scripts/predict_raw.py --weights /yol/ir_best.pt \
      --dataset_dir /content/dataset_ir --split val --modality ir \
      --out /yol/dets/ir_val.json.gz
"""

import argparse
import math
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eoir.dets import save_dets  # noqa: E402


def val_like_imgsz(h: int, w: int, imgsz: int = 640, stride: int = 32, pad: float = 0.5):
    """Ultralytics BaseDataset.set_rectangle ile ayni (h, w) girdi boyutu."""
    ar = h / w
    shape = [ar, 1.0] if ar < 1 else ([1.0, 1.0 / ar] if ar > 1 else [1.0, 1.0])
    return tuple(int(math.ceil(s * imgsz / stride + pad) * stride) for s in shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--modality", required=True, choices=["eo", "ir"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--half", action="store_true", help="fp16 cikarim (yalnizca GPU)")
    args = parser.parse_args()

    import ultralytics
    from ultralytics import YOLO

    img_paths = sorted((args.dataset_dir / "images" / args.split).glob("*.jpg"))
    if not img_paths:
        raise SystemExit(f"Goruntu yok: {args.dataset_dir / 'images' / args.split}")

    sizes = set()
    for p in img_paths:
        with Image.open(p) as im:
            sizes.add(im.size)
    if len(sizes) != 1:
        raise SystemExit(f"Goruntu boyutlari farkli ({sizes}); val ile ayni girdi boyutu tek degil.")
    (w, h), = sizes
    imgsz = val_like_imgsz(h, w, args.imgsz)
    print(f"{len(img_paths)} goruntu, boyut {w}x{h}, girdi (h, w) = {imgsz}")

    model = YOLO(str(args.weights))
    images = {}
    for k in range(0, len(img_paths), args.batch):
        chunk = [str(p) for p in img_paths[k:k + args.batch]]
        results = model.predict(chunk, imgsz=imgsz, conf=args.conf, iou=args.iou,
                                max_det=args.max_det, half=args.half, verbose=False)
        for path, r in zip(chunk, results):
            b = r.boxes
            images[Path(path).stem] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy())
        if (k // args.batch) % 20 == 0:
            print(f"  {min(k + args.batch, len(img_paths))}/{len(img_paths)}")

    meta = {
        "modality": args.modality, "split": args.split, "weights": args.weights.name,
        "conf": args.conf, "iou": args.iou, "max_det": args.max_det, "imgsz_hw": list(imgsz),
        "half": args.half, "image_wh": [w, h], "ultralytics": ultralytics.__version__,
        "box_format": "xyxy, orijinal piksel",
    }
    save_dets(args.out, images, meta)
    n = sum(len(s) for _, s in images.values())
    print(f"Yazildi: {args.out} ({len(images)} goruntu, {n} kutu)")


if __name__ == "__main__":
    main()
