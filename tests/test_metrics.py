"""Degerlendirici testleri: bilinen cevabi olan sentetik durumlar."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eoir.metrics import box_iou, evaluate, greedy_tp_labels, match_predictions  # noqa: E402


def random_gt(n_images=50, seed=0):
    rng = np.random.default_rng(seed)
    gt = {}
    for k in range(n_images):
        n = rng.integers(1, 6)
        xy = rng.uniform(0, 1000, (n, 2))
        wh = rng.uniform(40, 200, (n, 2))
        gt[f"05{k:04d}"] = np.hstack([xy, xy + wh])
    return gt


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-6)


def test_perfect_detections_hit_ultralytics_ceiling():
    """Kusursuz tespit 101 noktali interpolasyonda 0,995 verir (1,0 degil)."""
    gt = random_gt()
    dets = {s: (b.copy(), np.linspace(0.99, 0.9, len(b))) for s, b in gt.items()}
    r = evaluate(gt, dets)
    assert r["map50"] == pytest.approx(0.995, abs=1e-9)
    assert r["map50_95"] == pytest.approx(0.995, abs=1e-9)
    assert r["precision"] == pytest.approx(1.0, abs=1e-6)
    assert r["recall"] == pytest.approx(1.0, abs=1e-6)


def test_shift_hurts_map50_95_not_map50():
    """Kutulari kaydirmak gevsek esigi korur, siki esikleri bozar."""
    gt = random_gt()
    dets = {s: (b + 8.0, np.full(len(b), 0.9)) for s, b in gt.items()}
    r = evaluate(gt, dets)
    assert r["map50"] == pytest.approx(0.995, abs=1e-9)
    assert r["map50_95"] < 0.9


def test_missing_half_the_objects_halves_recall():
    gt = random_gt()
    dets = {s: (b[: (len(b) + 1) // 2], np.full((len(b) + 1) // 2, 0.9)) for s, b in gt.items()}
    n_gt = sum(len(b) for b in gt.values())
    n_found = sum((len(b) + 1) // 2 for b in gt.values())
    r = evaluate(gt, dets)
    assert r["recall"] == pytest.approx(n_found / n_gt, abs=2e-3)


def test_low_score_false_alarms_do_not_change_ap_much():
    """Tum dogrulardan dusuk skorlu yanlis alarmlar AP'yi bozmaz (siralama korunur)."""
    gt = random_gt()
    dets = {}
    for s, b in gt.items():
        fp = np.array([[2000, 2000, 2050, 2050]] * 5, dtype=float)
        dets[s] = (np.vstack([b, fp]), np.concatenate([np.full(len(b), 0.9), np.full(5, 0.01)]))
    assert evaluate(gt, dets)["map50"] == pytest.approx(0.995, abs=1e-9)


def test_duplicate_detection_counts_once():
    gt = {"050000": np.array([[0, 0, 100, 100]], dtype=float)}
    boxes = np.array([[0, 0, 100, 100], [1, 1, 101, 101]], dtype=float)
    tp = match_predictions(box_iou(gt["050000"], boxes))
    assert tp[:, 0].sum() == 1
    assert greedy_tp_labels(gt["050000"], boxes, np.array([0.5, 0.9])).tolist() == [0, 1]


def test_images_without_detections_still_count_their_people():
    gt = random_gt()
    dets = {s: (b, np.full(len(b), 0.9)) for s, b in gt.items()}
    half = {s: v for k, (s, v) in enumerate(dets.items()) if k % 2 == 0}
    r_all, r_half = evaluate(gt, dets), evaluate(gt, half)
    assert r_half["n_gt"] == r_all["n_gt"]
    assert r_half["recall"] < 0.7
