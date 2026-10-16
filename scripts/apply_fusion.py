"""
Dondurulmus fuzyonu uygulama
============================

tune_fusion.py'nin yazdigi fusion_config.json (ya da wbf_config.json,
single_config.json) ile EO ve IR tespitlerini birlestirip predict_raw bicimindeki bir dosyaya
yazar. Hicbir sey ogrenmez ve metrik hesaplamaz, yani test tespitlerine
guvenle uygulanabilir. Olcum evaluate_dets.py ile yapilir.

Kullanim:
  python3 scripts/apply_fusion.py --config results/fusion_val/fusion_config.json \
      --eo dets/eo_test.json.gz --ir dets/ir_test.json.gz --out dets/fusion_test.json.gz
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eoir.dets import load_dets, save_dets  # noqa: E402
from eoir.fusion import FusionModel, single_sensor_nms, wbf_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--eo", required=True, type=Path)
    parser.add_argument("--ir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    eo, eo_meta = load_dets(args.eo)
    ir, ir_meta = load_dets(args.ir)
    if eo_meta.get("split") != ir_meta.get("split"):
        raise SystemExit(f"EO ({eo_meta.get('split')}) ve IR ({ir_meta.get('split')}) farkli bolumlerden")
    if sorted(eo) != sorted(ir):
        raise SystemExit("EO ve IR dosyalari ayni goruntuleri kapsamiyor")
    if eo_meta.get("modality") == ir_meta.get("modality") and eo_meta.get("modality"):
        raise SystemExit("--eo ve --ir ayni modaliteden geliyor, dosyalari kontrol et")

    if cfg["method"] == "fusion_model":
        fused = FusionModel.from_dict(cfg["model"]).fuse(eo, ir)
        desc = cfg["model"]["config"]
    elif cfg["method"] == "wbf":
        wh = eo_meta.get("image_wh") or ir_meta.get("image_wh")
        fused = wbf_reference(eo, ir, wh, iou_thr=cfg["iou_thr"], skip_box_thr=cfg["skip_box_thr"],
                              max_det=cfg["max_det"])
        desc = {"iou_thr": cfg["iou_thr"]}
    elif cfg["method"] == "single_nms":
        # Karsilastirma noktasi: tek sensor + val'de secilen son NMS
        fused = single_sensor_nms(eo if cfg["modality"] == "eo" else ir, cfg["final_nms"], cfg["max_det"])
        desc = {"modality": cfg["modality"], "final_nms": cfg["final_nms"]}
    else:
        raise SystemExit(f"Bilinmeyen yontem: {cfg['method']}")

    meta = {"modality": "fusion", "split": eo_meta.get("split"), "method": cfg["method"], "config": desc,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "inputs": [str(args.eo), str(args.ir)], "image_wh": eo_meta.get("image_wh")}
    save_dets(args.out, fused, meta)
    print(f"Yazildi: {args.out} ({len(fused)} goruntu, {sum(len(s) for _, s in fused.values())} kutu)")


if __name__ == "__main__":
    main()
