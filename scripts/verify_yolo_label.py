"""
YOLO etiket dogrulama araci
============================
Belirtilen bir goruntu + YOLO .txt etiket ciftini okuyup, kutuyu
goruntunun uzerine cizer ve ekrana gosterir. Amac: donusum pipeline'inin
matematiksel olarak dogru calistigini GOZLE dogrulamak.

Kullanim:
  python3 scripts/verify_yolo_label.py --dataset_dir /path/to/output_v2/dataset_ir --split train --name 010001
"""

import argparse
import cv2
from pathlib import Path


def yolo_line_to_pixel_box(line, img_width, img_height):
    """
    'class_id x_center y_center w h' (hepsi 0-1 normalize) formatindaki
    bir YOLO satirini, cizim icin piksel koordinatlarina geri cevirir.
    Bu, script'in bir onceki asamada yaptigi donusumun TAM TERSI --
    eger ileri donusum dogruysa, bu geri donusum orijinal kutuyu vermeli.
    """
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center, y_center, box_w, box_h = map(float, parts[1:])

    # normalize -> mutlak piksel
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    box_w_px = box_w * img_width
    box_h_px = box_h * img_height

    # merkez-tabanli -> kose-tabanli (cizim icin xmin,ymin,xmax,ymax lazim)
    xmin = int(x_center_px - box_w_px / 2)
    ymin = int(y_center_px - box_h_px / 2)
    xmax = int(x_center_px + box_w_px / 2)
    ymax = int(y_center_px + box_h_px / 2)

    return class_id, xmin, ymin, xmax, ymax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=Path,
                         help="orn. output/dataset_ir")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--name", required=True, help="dosya adi, uzantisiz. orn. 260536")
    args = parser.parse_args()

    img_path = args.dataset_dir / "images" / args.split / f"{args.name}.jpg"
    lbl_path = args.dataset_dir / "labels" / args.split / f"{args.name}.txt"

    if not img_path.exists():
        raise FileNotFoundError(f"Goruntu bulunamadi: {img_path}")
    if not lbl_path.exists():
        raise FileNotFoundError(f"Etiket bulunamadi: {lbl_path}")

    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]

    lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    print(f"{len(lines)} kutu bulundu, goruntu boyutu: {img_width}x{img_height}")

    for line in lines:
        class_id, xmin, ymin, xmax, ymax = yolo_line_to_pixel_box(line, img_width, img_height)
        print(f"  -> class={class_id}, kose koordinatlari: ({xmin},{ymin}) - ({xmax},{ymax})")
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    out_path = args.dataset_dir / f"_verify_{args.name}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"\nKutulu goruntu kaydedildi -> {out_path}")
    print("Bu dosyayi acip, kirmizi kutunun gercekten kisinin uzerinde olup olmadigini goz ile kontrol et.")


if __name__ == "__main__":
    main()
