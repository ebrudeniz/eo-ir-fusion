"""
LLVIP -> YOLO format donusturucu (v2: klip bazli train/val bolunmesi)
=====================================================================

v1'den fark: val kumesi artik KARE bazinda degil KLIP bazinda ayriliyor.
Dosya adlari "KKFFFF" seklinde; KK = klip numarasi, FFFF = kare sirasi.
check_split_leakage.py ciktisi bunu destekledi (ayni klibin kareleri
ardisik numarali, val karelerinin %97'si train'de 1 kare mesafede).
Bir klip ya tamamen train'e ya tamamen val'e gider.

Cikti:
  dataset_eo/ ve dataset_ir/  ->  images/{train,val,test}, labels/{train,val,test}
  split_manifest.json         ->  hangi klipler val'e gitti (tekrarlanabilirlik)

Kullanim:
  python3 scripts/prepare_llvip_yolo.py --llvip_root /path/to/LLVIP --out_root /path/to/NEW_output

DIKKAT: --out_root icinde dataset_eo veya dataset_ir zaten varsa script durur.
Eski bolunmeyle yeni bolunme ayni klasorde karisirsa bir goruntu hem train'de
hem val'de kalir, yani sizinti geri gelir.
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

CLASS_TO_ID = {"person": 0}
CLIP_DIGITS = 2  # "010001" -> klip "01"


def parse_voc_xml(xml_path: Path):
    """Tek bir VOC XML'den (width, height, objects) dondurur."""
    root = ET.parse(xml_path).getroot()

    size_node = root.find("size")
    width = int(size_node.find("width").text)
    height = int(size_node.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_TO_ID:
            print(f"  [uyari] Bilinmeyen sinif '{name}' -> {xml_path.name}, obje atlandi.")
            continue

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Koordinatlar goruntu sinirlarini asmasin.
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            print(f"  [uyari] Gecersiz kutu atlandi -> {xml_path.name}")
            continue

        objects.append((name, xmin, ymin, xmax, ymax))

    return width, height, objects


def voc_box_to_yolo_line(class_name, xmin, ymin, xmax, ymax, width, height):
    """Kose-tabanli piksel kutusunu 'sinif x_merkez y_merkez w h' (0-1) satirina cevirir."""
    class_id = CLASS_TO_ID[class_name]
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def convert_all_annotations(annotations_dir: Path, labels_out_dir: Path):
    """Tum XML'leri YOLO .txt'ye cevirir. Etiketsiz kalan dosyalari ayrica dondurur."""
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"'{annotations_dir}' icinde .xml dosyasi bulunamadi.")

    print(f"{len(xml_files)} adet XML etiket donusturuluyor...")

    stems, empty_stems = [], []
    for xml_path in xml_files:
        width, height, objects = parse_voc_xml(xml_path)
        lines = [
            voc_box_to_yolo_line(name, xmin, ymin, xmax, ymax, width, height)
            for (name, xmin, ymin, xmax, ymax) in objects
        ]
        (labels_out_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stems.append(xml_path.stem)
        if not lines:
            empty_stems.append(xml_path.stem)

    print(f"Donusum tamamlandi. Etiketsiz kalan dosya sayisi: {len(empty_stems)}")
    return stems, empty_stems


def split_train_val_by_clip(train_stems, val_ratio=0.15, seed=42):
    """
    Train havuzunu KLIP bazinda train/val'e boler.
    Klipler seed ile karistirilir, hedef val boyutuna yaklasana kadar val'e eklenir.
    Hedefin %125'ini asacak buyuk bir klip varsa atlanir.
    """
    clips = defaultdict(list)
    for stem in train_stems:
        clips[stem[:CLIP_DIGITS]].append(stem)

    clip_ids = sorted(clips)
    random.Random(seed).shuffle(clip_ids)

    target = len(train_stems) * val_ratio
    val_stems, val_clips = set(), []
    for clip_id in clip_ids:
        size = len(clips[clip_id])
        if len(val_stems) + size > target * 1.25:
            continue
        val_stems.update(clips[clip_id])
        val_clips.append(clip_id)
        if len(val_stems) >= target * 0.9:
            break

    train_final = set(train_stems) - val_stems
    return train_final, val_stems, sorted(val_clips)


def build_modality_dataset(modality_name, llvip_root, out_root, all_labels_dir,
                           train_stems, val_stems, test_stems):
    """Tek modalite (infrared/visible) icin images/ ve labels/ klasorlerini doldurur."""
    src_train_dir = llvip_root / modality_name / "train"
    src_test_dir = llvip_root / modality_name / "test"
    out_dataset_dir = out_root / f"dataset_{'ir' if modality_name == 'infrared' else 'eo'}"

    split_map = {
        "train": (train_stems, src_train_dir),
        "val": (val_stems, src_train_dir),
        "test": (test_stems, src_test_dir),
    }

    for split_name, (stems, src_dir) in split_map.items():
        img_out_dir = out_dataset_dir / "images" / split_name
        lbl_out_dir = out_dataset_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        missing = 0
        for stem in sorted(stems):
            src_img = src_dir / f"{stem}.jpg"
            src_lbl = all_labels_dir / f"{stem}.txt"
            if not src_img.exists() or not src_lbl.exists():
                missing += 1
                continue
            shutil.copy2(src_img, img_out_dir / f"{stem}.jpg")
            shutil.copy2(src_lbl, lbl_out_dir / f"{stem}.txt")

        print(f"  [{modality_name}/{split_name}] {len(stems) - missing} dosya kopyalandi"
              f"{f', {missing} eksik atlandi' if missing else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvip_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Eski ciktinin ustune yazmayi engelle (sizintiyi geri getirir).
    for name in ("dataset_ir", "dataset_eo"):
        if (args.out_root / name).exists():
            raise SystemExit(
                f"'{args.out_root / name}' zaten var. Yeni bir --out_root ver "
                f"ya da eski klasoru once sil."
            )

    annotations_dir = args.llvip_root / "Annotations"
    infrared_train_dir = args.llvip_root / "infrared" / "train"
    infrared_test_dir = args.llvip_root / "infrared" / "test"

    tmp_labels_dir = args.out_root / "_all_labels_tmp"
    all_stems, empty_stems = convert_all_annotations(annotations_dir, tmp_labels_dir)
    all_stems_set = set(all_stems)

    train_pool = {p.stem for p in infrared_train_dir.glob("*.jpg")} & all_stems_set
    test_pool = {p.stem for p in infrared_test_dir.glob("*.jpg")} & all_stems_set
    print(f"\nLLVIP train havuzu: {len(train_pool)} sahne | test havuzu: {len(test_pool)} sahne")

    train_stems, val_stems, val_clips = split_train_val_by_clip(
        train_pool, args.val_ratio, args.seed
    )
    ratio = len(val_stems) / len(train_pool)
    print(f"Val'e giden klipler: {val_clips}")
    print(f"-> train: {len(train_stems)} | val: {len(val_stems)} ({ratio:.1%}) | test: {len(test_pool)}\n")

    for modality in ("infrared", "visible"):
        print(f"'{modality}' dataseti olusturuluyor...")
        build_modality_dataset(
            modality, args.llvip_root, args.out_root, tmp_labels_dir,
            train_stems, val_stems, test_pool,
        )

    manifest = {
        "seed": args.seed,
        "val_ratio_target": args.val_ratio,
        "val_clips": val_clips,
        "n_train": len(train_stems),
        "n_val": len(val_stems),
        "n_test": len(test_pool),
        "empty_label_stems": empty_stems,
    }
    (args.out_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    shutil.rmtree(tmp_labels_dir)
    print("\nTamamlandi. split_manifest.json yazildi.")
    if empty_stems:
        print(f"Etiketsiz kalan dosyalar (gorselleri kontrol et): {empty_stems}")


if __name__ == "__main__":
    main()
