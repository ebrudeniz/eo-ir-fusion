"""
Tespit degerlendirici
=====================

predict_raw bicimindeki bir ya da birden cok tespit dosyasini ayni etiketlere
karsi olcer (eoir/metrics.py, Ultralytics 8.4.131 ile ayni hesap).
Baseline'lar ve fuzyon hep bu betikle olculur.

Ornekler:
  # IR ve EO val, Ultralytics'in verdigi sayilarla karsilastir (saglama)
  python3 scripts/evaluate_dets.py --dataset_dir /yol/dataset_ir --split val \
      --dets ir=dets/ir_val.json.gz eo=dets/eo_val.json.gz \
      --expect ir=0.958,0.616 eo=0.877,0.492

  # Klip bazli ve eslestirilmis bootstrap ile karsilastirma
  python3 scripts/evaluate_dets.py ... --per_clip --bootstrap 1000 --reference ir

Test korumasi: --split test yalnizca --final ile calisir ve her calisma
results/test_eval_log.jsonl'e (dosya ozetleriyle) yazilir. Ayni tespit
dosyasi ikinci kez olculmek istenirse betik durur. Amac: test'e bakip
yontemi ayarlamayi zorlastirmak ve kac kez bakildigini kayit altina almak.
Etiketler EO ve IR'da ayni oldugu icin --dataset_dir ikisinden biri olabilir.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eoir.dets import check_same_images, clip_of, load_dets, load_gt  # noqa: E402
from eoir.metrics import image_stats, paired_bootstrap, summarize  # noqa: E402

TEST_LOG = ROOT / "results" / "test_eval_log.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_named(items, what):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{what} 'isim=deger' biciminde olmali: {it}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def guard_test(det_paths: dict, force_again: bool):
    hashes = {name: sha256(p) for name, p in det_paths.items()}
    seen = set()
    if TEST_LOG.exists():
        for line in TEST_LOG.read_text(encoding="utf-8").splitlines():
            seen.update(json.loads(line)["sha256"].values())
    again = [n for n, h in hashes.items() if h in seen]
    if again and not force_again:
        raise SystemExit(
            f"Bu tespit dosyalari test'te zaten olculmus: {again}. Sonuc {TEST_LOG} icinde. "
            "Gercekten tekrar olcmek istiyorsan --force_again ver; bu da kayda gecer."
        )
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--dets", required=True, nargs="+", help="isim=yol ...")
    parser.add_argument("--expect", nargs="*", default=[],
                        help="isim=mAP50,mAP50-95 (saglama icin beklenen degerler)")
    parser.add_argument("--tol", type=float, default=0.005, help="--expect icin izin verilen sapma")
    parser.add_argument("--per_clip", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--reference", help="bootstrap farklari bu yonteme gore")
    parser.add_argument("--final", action="store_true", help="test'i olcmek icin zorunlu")
    parser.add_argument("--force_again", action="store_true")
    parser.add_argument("--note", default="", help="test kaydina eklenecek aciklama")
    parser.add_argument("--out", type=Path, help="sonuclari JSON olarak yaz")
    args = parser.parse_args()

    det_paths = {k: Path(v) for k, v in parse_named(args.dets, "--dets").items()}
    if args.split == "test":
        if not args.final:
            raise SystemExit(
                "Test bolumu yalnizca --final ile olculur. Once tum kararlari val'de dondur "
                "(fusion_config.json), sonra baseline'lar ve fuzyonu TEK SEFERDE olc."
            )
        hashes = guard_test(det_paths, args.force_again)

    gt = load_gt(args.dataset_dir, args.split)
    stats, metas = {}, {}
    for name, path in det_paths.items():
        dets, meta = load_dets(path)
        if meta.get("split") not in (None, args.split):
            raise SystemExit(f"{name}: dosya '{meta.get('split')}' bolumunden, istenen '{args.split}'")
        check_same_images(gt, dets, name)
        stats[name], metas[name] = image_stats(gt, dets), meta

    report = {"split": args.split, "n_images": len(gt), "n_gt": int(sum(len(b) for b in gt.values())),
              "methods": {}}
    print(f"\n{args.split}: {report['n_images']} goruntu, {report['n_gt']} insan\n")
    print(f"{'yontem':24s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s} {'kutu':>8s}")
    for name, st in stats.items():
        r = summarize(st)
        report["methods"][name] = {"overall": r, "file": str(det_paths[name]), "meta": metas[name]}
        print(f"{name:24s} {r['precision']:6.3f} {r['recall']:6.3f} {r['map50']:7.3f} "
              f"{r['map50_95']:9.3f} {r['n_dets']:8d}")

    if args.per_clip:
        clips = sorted({clip_of(s) for s in gt})
        print("\nKlip bazli mAP50-95 (mAP50)")
        print(f"{'yontem':24s} " + " ".join(f"{c:>13s}" for c in clips))
        for name, st in stats.items():
            row = {}
            for c in clips:
                r = summarize(st, [s for s in st if clip_of(s) == c])
                row[c] = {"map50": r["map50"], "map50_95": r["map50_95"], "n_gt": r["n_gt"]}
            report["methods"][name]["per_clip"] = row
            print(f"{name:24s} " + " ".join(
                f"{row[c]['map50_95']:6.3f}({row[c]['map50']:.3f})" for c in clips))

    if args.bootstrap:
        ref = args.reference or next(iter(stats))
        boot = paired_bootstrap(stats, args.bootstrap)
        print(f"\nEslestirilmis bootstrap ({args.bootstrap} ornek, goruntu bazli), fark = yontem - {ref}")
        print("Not: ayni klipteki kareler bagimsiz degil, bu aralik gercek belirsizligi kucuk gosterir.")
        for name in stats:
            if name == ref:
                continue
            out = {}
            for k in ("map50", "map50_95"):
                d = boot[name][k] - boot[ref][k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                out[k] = {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                          "p_diff_le_0": float((d <= 0).mean())}
                print(f"  {name:22s} {k:9s} fark {d.mean():+.4f}  %95 aralik [{lo:+.4f}, {hi:+.4f}]")
            report["methods"][name]["bootstrap_vs_" + ref] = out

    failed = []
    for name, val in parse_named(args.expect, "--expect").items():
        e50, e5095 = (float(v) for v in val.split(","))
        r = report["methods"][name]["overall"]
        d50, d5095 = r["map50"] - e50, r["map50_95"] - e5095
        ok = abs(d50) <= args.tol and abs(d5095) <= args.tol
        print(f"\nSaglama {name}: mAP50 {r['map50']:.4f} (beklenen {e50}, fark {d50:+.4f}), "
              f"mAP50-95 {r['map50_95']:.4f} (beklenen {e5095}, fark {d5095:+.4f}) -> "
              f"{'TAMAM' if ok else 'TUTMUYOR'}")
        if not ok:
            failed.append(name)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nYazildi: {args.out}")

    if args.split == "test":
        TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": args.note, "sha256": hashes,
                 "overall": {n: {k: m["overall"][k] for k in ("precision", "recall", "map50", "map50_95")}
                             for n, m in report["methods"].items()}}
        with open(TEST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Test olcumu kayda gecti: {TEST_LOG}")

    if failed:
        raise SystemExit(f"Saglama tutmadi: {failed}. Fuzyona gecmeden olcum farkini bul.")


if __name__ == "__main__":
    main()
