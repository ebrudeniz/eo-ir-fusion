"""
Val sizintisi kontrolu
======================
Hipotez: LLVIP dosya adlari "KKFFFF" seklinde. KK = cekim/klip numarasi,
FFFF = o klipteki kare sirasi. Bu hipotez DOGRU OLMAYABILIR; script bunu
dosya adlarindan gozlemler, kanitlamaz. Ciktiya birlikte bakacagiz.

Sorular:
  1. Hangi KK degerleri train/val/test icinde yer aliyor? (klipler bolunmus mu?)
  2. Her val karesinin, ayni klipteki en yakin train karesine uzakligi ne kadar?
     Uzaklik 1-2 kare ise val'deki kare train'dekinin neredeyse kopyasidir.

Kullanim:
  python3 scripts/check_split_leakage.py --dataset_dir output_v2/dataset_ir
"""

import argparse
import bisect
import statistics
from collections import defaultdict
from pathlib import Path


def list_stems(dataset_dir, split):
    return sorted(p.stem for p in (dataset_dir / "images" / split).glob("*.jpg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=Path)
    args = parser.parse_args()

    splits = {s: list_stems(args.dataset_dir, s) for s in ("train", "val", "test")}

    # 1) Klip numarasi (ilk 2 hane) bazinda dagilim
    counts = {s: defaultdict(int) for s in splits}
    for s, stems in splits.items():
        for stem in stems:
            counts[s][stem[:2]] += 1

    prefixes = sorted({k for s in counts.values() for k in s})
    print("KK     train    val   test")
    for k in prefixes:
        print(f"{k}   {counts['train'][k]:6d} {counts['val'][k]:6d} {counts['test'][k]:6d}")

    # 1b) Hangi klipler birden fazla kumede gorunuyor?
    clips = {s: {stem[:2] for stem in stems} for s, stems in splits.items()}
    print()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(clips[a] & clips[b])
        print(f"{a} ve {b} ortak klipler: {shared if shared else 'yok'}")

    # 2) Her val karesi icin ayni klipteki en yakin train karesine uzaklik
    train_frames = defaultdict(list)
    for stem in splits["train"]:
        train_frames[stem[:2]].append(int(stem[2:]))
    for frames in train_frames.values():
        frames.sort()

    distances = []
    for stem in splits["val"]:
        frames = train_frames.get(stem[:2])
        if not frames:
            continue
        f = int(stem[2:])
        i = bisect.bisect_left(frames, f)
        candidates = []
        if i < len(frames):
            candidates.append(abs(frames[i] - f))
        if i > 0:
            candidates.append(abs(frames[i - 1] - f))
        distances.append(min(candidates))

    print(f"\nVal karesi sayisi (ayni klipte train karesi olan): {len(distances)}")
    if distances:
        n = len(distances)
        print(f"En yakin train karesine uzaklik: medyan = {statistics.median(distances)}")
        for limit in (1, 5, 20):
            share = sum(d <= limit for d in distances) / n
            print(f"  uzaklik <= {limit:2d} kare olan val orani: {share:.1%}")


if __name__ == "__main__":
    main()
