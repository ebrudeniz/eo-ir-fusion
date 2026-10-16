"""
Fuzyon ayarlarini val'de klip bazli capraz dogrulamayla secme
=============================================================

Sorun: val iki kez kullaniliyor (en iyi epoch secimi + fuzyon ayari).
Fuzyonun ogrenilen parcalari (kalibrasyon, lojistik regresyon) ayni
karelerde hem ogrenilip hem olculurse skor iyimser cikar. Bu yuzden:

  - 4 val klibinin her biri sirayla disarida birakilir (leave-one-clip-out).
  - Kalibrasyon ve ogrenilmis fuzyon kalan 3 klipte ogrenilir, disaridaki
    klibe uygulanir. Her karenin fuzyon ciktisi, o kareyi hic gormemis bir
    modelden gelir (out-of-fold, OOF).
  - Tum OOF ciktilari birlestirilip tek bir val mAP'i hesaplanir.
  - Birincil olcut ONCEDEN secildi: OOF mAP50-95 (Ultralytics'in fitness'i
    da bu). Yalnizca bu olcute gore secim yapilir.
  - Secilen ayar tum val'de yeniden ogrenilir ve fusion_config.json'a
    dondurulur. Test'te yalnizca bu dosya kullanilir.

Dikkat: onlarca ayar arasindan en iyisini secmek OOF skorunu da hafif
iyimser yapar (secim yanliligi). Durust sayi test'ten gelir; burada
ayrica "en iyi ayar kac klipte IR'i geciyor" gibi kararlilik bilgisi raporlanir.

Kullanim:
  python3 scripts/tune_fusion.py --dataset_dir /yol/dataset_ir \
      --eo dets/eo_val.json.gz --ir dets/ir_val.json.gz --out_dir results/fusion_val
"""

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eoir.dets import check_same_images, clip_of, load_dets, load_gt  # noqa: E402
from eoir.fusion import (BOX_RULES, SCORE_RULES, Calibrator, FusionModel,  # noqa: E402
                         calibration_data, expected_calibration_error, wbf_reference)
from eoir.metrics import image_stats, summarize  # noqa: E402

MATCH_IOUS = (0.3, 0.5, 0.7)
WBF_IOUS = (0.4, 0.55, 0.7)
PRIMARY = "map50_95"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(stats, clips):
    out = {"overall": summarize(stats)}
    for c in clips:
        out[c] = summarize(stats, [s for s in stats if clip_of(s) == c])
    return out


def row_of(name, cfg, res, clips):
    row = {"method": name, **cfg,
           "map50": res["overall"]["map50"], "map50_95": res["overall"]["map50_95"],
           "precision": res["overall"]["precision"], "recall": res["overall"]["recall"]}
    for c in clips:
        row[f"clip{c}_map50_95"] = res[c]["map50_95"]
        row[f"clip{c}_map50"] = res[c]["map50"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--eo", required=True, type=Path)
    parser.add_argument("--ir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--skip_wbf", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    gt = load_gt(args.dataset_dir, "val")
    eo, eo_meta = load_dets(args.eo)
    ir, ir_meta = load_dets(args.ir)
    for name, d, meta in (("eo", eo, eo_meta), ("ir", ir, ir_meta)):
        if meta.get("split") not in (None, "val"):
            raise SystemExit(f"{name}: tune_fusion yalnizca val tespitleriyle calisir (dosya: {meta.get('split')})")
        check_same_images(gt, d, name)
    stems = sorted(gt)
    clips = sorted({clip_of(s) for s in stems})
    folds = {c: ([s for s in stems if clip_of(s) != c], [s for s in stems if clip_of(s) == c]) for c in clips}
    print(f"val: {len(stems)} goruntu, klipler {clips}, birincil olcut OOF {PRIMARY}")

    rows, results = [], {}

    # Baseline'lar: ogrenilen parca yok, dogrudan olculur.
    for name, d in (("eo_only", eo), ("ir_only", ir)):
        res = scores(image_stats(gt, d), clips)
        results[name] = res
        rows.append(row_of(name, {}, res, clips))
        print(f"  {name:10s} mAP50 {res['overall']['map50']:.4f}  mAP50-95 {res['overall']['map50_95']:.4f}")

    # Fold basina kalibrasyon (tum fuzyon ayarlari ayni fold kalibrasyonunu paylasir).
    fold_cal, calib_report = {}, {}
    oof = {"eo": {"raw": [], "platt": [], "y": []}, "ir": {"raw": [], "platt": [], "y": []}}
    for c, (tr, te) in folds.items():
        cals = []
        for key, d in (("eo", eo), ("ir", ir)):
            cal = Calibrator("platt").fit(*calibration_data(gt, d, tr))
            s_te, y_te = calibration_data(gt, d, te)
            oof[key]["raw"].append(s_te)
            oof[key]["platt"].append(cal(s_te))
            oof[key]["y"].append(y_te)
            cals.append(cal)
        fold_cal[c] = tuple(cals)
    for key in ("eo", "ir"):
        y = np.concatenate(oof[key]["y"])
        calib_report[key] = {}
        for kind in ("raw", "platt"):
            ece, bins = expected_calibration_error(np.concatenate(oof[key][kind]), y)
            calib_report[key][kind] = {"ece": ece, "bins": bins}
        print(f"  kalibrasyon {key}: OOF ECE ham {calib_report[key]['raw']['ece']:.4f} "
              f"-> Platt {calib_report[key]['platt']['ece']:.4f}")

    # Fuzyon izgarasi. Eslestirme (pahali kisim) yalnizca fold kalibrasyonuna ve
    # match_iou'ya bagli; her (fold, match_iou) icin bir kez kurulup tum skor ve
    # kutu kurallarinda yeniden kullanilir. Sonuc degismez, yalnizca hizlanir.
    grid = list(itertools.product(MATCH_IOUS, SCORE_RULES, BOX_RULES))
    print(f"\n{len(grid)} fuzyon ayari x {len(clips)} fold")
    k = 0
    for miou in MATCH_IOUS:
        cache = {}
        for c in clips:
            builder = FusionModel(match_iou=miou)
            builder.cal_eo, builder.cal_ir = fold_cal[c]
            cache[c] = {s: builder.clusters(eo[s], ir[s]) for s in stems}
        for srule, brule in itertools.product(SCORE_RULES, BOX_RULES):
            k += 1
            cfg = {"match_iou": miou, "score_rule": srule, "box_rule": brule, "calibration": "platt"}
            fused = {}
            for c, (tr, te) in folds.items():
                m = FusionModel(**cfg).fit(gt, eo, ir, tr, calibrators=fold_cal[c], clusters=cache[c])
                fused.update(m.fuse(eo, ir, te, clusters=cache[c]))
            res = scores(image_stats(gt, fused), clips)
            name = f"fusion_{srule}_{brule}_m{miou}"
            results[name] = res
            rows.append(row_of(name, cfg, res, clips))
            print(f"  [{k:2d}/{len(grid)}] {name:34s} mAP50 {res['overall']['map50']:.4f}  "
                  f"mAP50-95 {res['overall']['map50_95']:.4f}", flush=True)
        del cache

    fusion_rows = [r for r in rows if r["method"].startswith("fusion_")]
    best = max(fusion_rows, key=lambda r: r[PRIMARY])
    best_cfg = {k: best[k] for k in ("match_iou", "score_rule", "box_rule", "calibration")}

    # Ablasyon: ayni ayar, kalibrasyonsuz (kalibrasyonun katkisi).
    abl_cfg = {**best_cfg, "calibration": "none"}
    if abl_cfg["score_rule"] != "learned":
        fused = {}
        for c, (tr, te) in folds.items():
            fused.update(FusionModel(**abl_cfg).fit(gt, eo, ir, tr).fuse(eo, ir, te))
        res = scores(image_stats(gt, fused), clips)
        results["ablation_best_without_calibration"] = res
        rows.append(row_of("ablation_best_without_calibration", abl_cfg, res, clips))

    # Literatur referansi: WBF (ogrenilen parca yok, yalnizca IoU esigi secilir).
    wbf_best = None
    if not args.skip_wbf:
        wh = eo_meta.get("image_wh") or ir_meta.get("image_wh")
        if wh is None:
            raise SystemExit("WBF icin tespit dosyasinda meta.image_wh gerekli (ya da --skip_wbf)")
        for thr in WBF_IOUS:
            res = scores(image_stats(gt, wbf_reference(eo, ir, wh, iou_thr=thr)), clips)
            name = f"wbf_iou{thr}"
            results[name] = res
            rows.append(row_of(name, {"wbf_iou_thr": thr}, res, clips))
            print(f"  {name:34s} mAP50 {res['overall']['map50']:.4f}  mAP50-95 {res['overall']['map50_95']:.4f}")
        wbf_rows = [r for r in rows if r["method"].startswith("wbf_")]
        wbf_best = max(wbf_rows, key=lambda r: r[PRIMARY])

    # Kararlilik: en iyi ayar her klipte IR'a gore ne yapiyor?
    ir_res, best_res = results["ir_only"], results[best["method"]]
    per_clip_delta = {c: best_res[c][PRIMARY] - ir_res[c][PRIMARY] for c in clips}

    # Son model: secilen ayar TUM val'de ogrenilir ve dondurulur.
    final = FusionModel(**best_cfg).fit(gt, eo, ir, stems)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"eo_val": {"file": str(args.eo), "sha256": sha256(args.eo)},
              "ir_val": {"file": str(args.ir), "sha256": sha256(args.ir)}}
    frozen = {"method": "fusion_model", "model": final.to_dict(), "selected_by": f"OOF {PRIMARY}",
              "oof_val": {k: best[k] for k in ("map50", "map50_95")}, "inputs": inputs,
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (args.out_dir / "fusion_config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    if wbf_best is not None:
        wbf_frozen = {"method": "wbf", "iou_thr": wbf_best["wbf_iou_thr"], "skip_box_thr": 0.0,
                      "max_det": 300, "selected_by": f"val {PRIMARY}", "inputs": inputs}
        (args.out_dir / "wbf_config.json").write_text(json.dumps(wbf_frozen, indent=2), encoding="utf-8")

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "method", k))
    with open(args.out_dir / "cv_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r[PRIMARY]))

    summary = {
        "primary_metric": f"OOF {PRIMARY}", "clips": clips, "n_configs": len(grid),
        "baselines": {n: results[n]["overall"] for n in ("eo_only", "ir_only")},
        "best_fusion": {"name": best["method"], "config": best_cfg, "oof": best_res["overall"]},
        "best_minus_ir_overall": {k: best_res["overall"][k] - ir_res["overall"][k] for k in ("map50", "map50_95")},
        "best_minus_ir_per_clip_map50_95": per_clip_delta,
        "clips_where_best_beats_ir": int(sum(v > 0 for v in per_clip_delta.values())),
        "wbf_best": None if wbf_best is None else {"name": wbf_best["method"], "map50": wbf_best["map50"],
                                                   "map50_95": wbf_best["map50_95"]},
        "calibration": calib_report, "runtime_s": round(time.time() - t0, 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_reliability(calib_report, args.out_dir / "reliability.png")

    d = summary["best_minus_ir_overall"]
    print(f"\nEn iyi fuzyon: {best['method']}  OOF mAP50 {best['map50']:.4f}  mAP50-95 {best['map50_95']:.4f}")
    print(f"IR'a gore fark: mAP50 {d['map50']:+.4f}  mAP50-95 {d['map50_95']:+.4f}  "
          f"(klip bazinda IR'i gectigi klip: {summary['clips_where_best_beats_ir']}/{len(clips)})")
    print(f"Yazildi: {args.out_dir}/fusion_config.json, cv_results.csv, summary.json")


def plot_reliability(calib_report, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, key in zip(axes, ("eo", "ir")):
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="ideal")
        for kind, color in (("raw", "#c0504d"), ("platt", "#1f4e79")):
            bins = calib_report[key][kind]["bins"]
            ax.plot([b["mean_p"] for b in bins], [b["frac_pos"] for b in bins], marker="o", ms=3,
                    color=color, label=f"{kind} (ECE {calib_report[key][kind]['ece']:.3f})")
        ax.set_title(f"{key.upper()} (val, out-of-fold)")
        ax.set_xlabel("tahmin edilen olasilik / skor")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("dogru tespit orani")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
