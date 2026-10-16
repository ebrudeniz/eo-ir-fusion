"""Fuzyon testleri: eslestirme, kurallar, kalibrasyon ve uctan uca sentetik deney."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eoir.dets import load_dets, save_dets  # noqa: E402
from eoir.fusion import Calibrator, FusionModel, match_pairs  # noqa: E402
from eoir.metrics import evaluate  # noqa: E402


def test_match_pairs_prefers_strong_boxes():
    """Guclu IR kutusu, EO'daki dusuk skorlu kopya yerine guclu EO kutusuyla eslesmeli."""
    ir = np.array([[0, 0, 100, 200]], dtype=float)
    eo = np.array([[2, 2, 102, 202], [0, 30, 100, 230]], dtype=float)  # iyi kutu, kayik kopya
    e, i, iou = match_pairs(eo, np.array([0.6, 0.05]), ir, np.array([0.9]), 0.5)
    pairs = {(a, b) for a, b in zip(e, i)}
    assert (0, 0) in pairs and (1, -1) in pairs


def test_match_pairs_one_to_one_and_complete():
    rng = np.random.default_rng(0)
    eo = rng.uniform(0, 500, (30, 2))
    eo = np.hstack([eo, eo + 50])
    ir = eo[:20] + rng.normal(0, 3, (20, 4))
    e, i, _ = match_pairs(eo, rng.uniform(size=30), ir, rng.uniform(size=20), 0.5)
    assert sorted(e[e >= 0].tolist()) == list(range(30))
    assert sorted(i[i >= 0].tolist()) == list(range(20))


def test_platt_recovers_monotone_calibration():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.001, 0.999, 20000)
    true_p = 1 / (1 + np.exp(-(2.0 * np.log(s / (1 - s)) - 1.0)))
    y = (rng.uniform(size=s.size) < true_p).astype(int)
    cal = Calibrator("platt").fit(s, y)
    assert cal.a == pytest.approx(2.0, abs=0.1)
    assert cal.b == pytest.approx(-1.0, abs=0.1)


def make_complementary_world(n_images=120, seed=0):
    """
    Her sensor insanlarin farkli bir kismini kaciriyor. Ideal bir fuzyon
    ikisinden de yuksek recall ve mAP vermeli. Klipler: 05, 06, 14, 15.
    """
    rng = np.random.default_rng(seed)
    gt, eo, ir = {}, {}, {}
    clips = ["05", "06", "14", "15"]
    for k in range(n_images):
        stem = f"{clips[k % 4]}{k:04d}"
        n = rng.integers(1, 5)
        xy = rng.uniform(0, 1000, (n, 2))
        g = np.hstack([xy, xy + rng.uniform(60, 200, (n, 2))])
        gt[stem] = g
        for store, miss_p, noise, fp_score in ((eo, 0.35, 6.0, 0.5), (ir, 0.15, 2.0, 0.3)):
            seen = rng.uniform(size=n) > miss_p
            b = g[seen] + rng.normal(0, noise, (seen.sum(), 4))
            s = rng.uniform(0.5, 0.95, seen.sum())
            fp_xy = rng.uniform(0, 1000, (2, 2))
            fp = np.hstack([fp_xy, fp_xy + 80])
            store[stem] = (np.vstack([b, fp]), np.concatenate([s, rng.uniform(0.05, fp_score, 2)]))
    return gt, eo, ir


@pytest.mark.parametrize("rule", ["max", "mean", "noisy_or", "learned"])
def test_fusion_beats_single_sensors_when_errors_are_complementary(rule):
    gt, eo, ir = make_complementary_world()
    m = FusionModel(match_iou=0.5, score_rule=rule, box_rule="ir").fit(gt, eo, ir, sorted(gt))
    fused = m.fuse(eo, ir)
    r_f, r_eo, r_ir = evaluate(gt, fused), evaluate(gt, eo), evaluate(gt, ir)
    assert r_f["map50"] > max(r_eo["map50"], r_ir["map50"])


def test_intersection_drops_single_sensor_boxes():
    gt, eo, ir = make_complementary_world()
    m = FusionModel(score_rule="intersection").fit(gt, eo, ir, sorted(gt))
    s = sorted(gt)[0]
    boxes, _ = m.fuse_image(eo[s], ir[s])
    assert len(boxes) <= min(len(eo[s][0]), len(ir[s][0]))


def test_max_det_cap():
    b = np.tile(np.array([[0, 0, 10, 10]], dtype=float), (300, 1)) + np.arange(300)[:, None] * 20
    s = np.linspace(0.9, 0.1, 300)
    m = FusionModel(score_rule="max", calibration="none")
    boxes, scores = m.fuse_image((b, s), (b + 5000, s))
    assert len(boxes) == 300 and scores.min() >= 0.1


def test_frozen_model_roundtrip():
    gt, eo, ir = make_complementary_world()
    m = FusionModel(score_rule="learned", box_rule="weighted").fit(gt, eo, ir, sorted(gt))
    m2 = FusionModel.from_dict(json.loads(json.dumps(m.to_dict())))
    s = sorted(gt)[3]
    np.testing.assert_allclose(m.fuse_image(eo[s], ir[s])[1], m2.fuse_image(eo[s], ir[s])[1])


def write_world(tmp_path, gt, eo, ir, split="val"):
    """Sentetik dunyayi gercek dosya bicimlerine yaz (YOLO etiket + jpg + dets)."""
    from PIL import Image

    W, H = 1280, 1024
    ds = tmp_path / "dataset_ir"
    (ds / "images" / split).mkdir(parents=True)
    (ds / "labels" / split).mkdir(parents=True)
    for stem, g in gt.items():
        Image.new("RGB", (W, H)).save(ds / "images" / split / f"{stem}.jpg")
        g = np.clip(g, 0, [W, H, W, H])
        lines = [f"0 {(a + c) / 2 / W:.6f} {(b + d) / 2 / H:.6f} {(c - a) / W:.6f} {(d - b) / H:.6f}"
                 for a, b, c, d in g]
        (ds / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        gt[stem] = g
    for name, d in (("eo", eo), ("ir", ir)):
        # Ultralytics predict kutulari goruntu sinirina kirpar; sentetik tespitler de oyle olmali.
        # (Kirpilmazsa sinira tasan kutular kirpilmis etikete gore yanlis alarm sayilir ve
        # kendi kirpan WBF yapay bir ustunluk kazanir.)
        d = {s: (np.clip(b, 0, [W, H, W, H]), sc) for s, (b, sc) in d.items()}
        save_dets(tmp_path / f"{name}_{split}.json.gz", d,
                  {"modality": name, "split": split, "image_wh": [W, H]})
    return ds


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)


def test_end_to_end_cli(tmp_path):
    """predict_raw sonrasi tum hat: tune (val) -> apply -> evaluate; test korumasi."""
    gt, eo, ir = make_complementary_world(n_images=60)
    ds = write_world(tmp_path, gt, eo, ir, "val")

    r = run(ROOT / "scripts/evaluate_dets.py", "--dataset_dir", ds, "--split", "val",
            "--dets", f"ir={tmp_path / 'ir_val.json.gz'}", f"eo={tmp_path / 'eo_val.json.gz'}", "--per_clip")
    assert r.returncode == 0, r.stderr

    out = tmp_path / "fusion_val"
    r = run(ROOT / "scripts/tune_fusion.py", "--dataset_dir", ds, "--eo", tmp_path / "eo_val.json.gz",
            "--ir", tmp_path / "ir_val.json.gz", "--out_dir", out)
    assert r.returncode == 0, r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["best_minus_single_overall"]["map50"] > 0
    assert (out / "single_config.json").exists()
    assert (out / "fusion_config.json").exists() and (out / "wbf_config.json").exists()

    r = run(ROOT / "scripts/apply_fusion.py", "--config", out / "fusion_config.json",
            "--eo", tmp_path / "eo_val.json.gz", "--ir", tmp_path / "ir_val.json.gz",
            "--out", tmp_path / "fusion_val.json.gz")
    assert r.returncode == 0, r.stderr
    fused, meta = load_dets(tmp_path / "fusion_val.json.gz")
    assert meta["split"] == "val" and len(fused) == len(gt)

    # Test korumasi: --final olmadan test olculemez.
    write_world(tmp_path / "t", *make_complementary_world(n_images=8, seed=1), split="test")
    r = run(ROOT / "scripts/evaluate_dets.py", "--dataset_dir", tmp_path / "t" / "dataset_ir", "--split", "test",
            "--dets", f"ir={tmp_path / 't' / 'ir_test.json.gz'}")
    assert r.returncode != 0 and "--final" in (r.stderr + r.stdout)


def test_final_nms_removes_same_sensor_duplicates():
    """Ayni sensordeki kopya kutu (IoU 0,6) son NMS 0,55 ile elenmeli, 0,7 NMS'ten kurtulmus olsa da."""
    from eoir.metrics import box_iou

    b = np.array([[0, 0, 100, 200], [0, 50, 100, 250]], dtype=float)  # IoU = 150/250 = 0,6
    assert box_iou(b[:1], b[1:])[0, 0] == pytest.approx(0.6, abs=1e-6)
    eo = (b, np.array([0.8, 0.3]))
    ir = (b[:1] + 1, np.array([0.9]))
    plain = FusionModel(score_rule="max", calibration="none").fuse_image(eo, ir)
    with_nms = FusionModel(score_rule="max", calibration="none", final_nms=0.55).fuse_image(eo, ir)
    assert len(plain[0]) == 2 and len(with_nms[0]) == 1
