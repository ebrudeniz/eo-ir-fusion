"""
Ultralytics 8.4.131 ile ayni sonucu veren tespit olcumu
=======================================================

Neden kendi kodumuz: baseline'lar ve fuzyon ayni olcum aletiyle olculmeli.
Biri Ultralytics ile, digeri bizim kodumuzla olculurse fark fuzyondan mi
olcumden mi geliyor ayrilamaz. Bu yuzden asagidaki fonksiyonlar Ultralytics
8.4.131'in su fonksiyonlarini birebir izliyor:

  engine/validator.py   BaseValidator.match_predictions  (use_scipy=False)
  utils/metrics.py      box_iou, compute_ap, ap_per_class, smooth

Iki asama:
  1) image_stats: her goruntu icin her tespitin 10 IoU esiginde dogru mu
     (tp[N,10]), skoru ve goruntudeki gercek kutu sayisi. Bu asama pahali.
  2) summarize: istatistikleri birlestirip P, R, mAP50, mAP50-95 hesaplar.
     Ucuz, bu yuzden bootstrap 1. asamayi tekrar etmeden 2.'yi tekrarlar.
"""

import numpy as np

IOUV = np.linspace(0.5, 0.95, 10)
EPS = 1e-16


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a[N,4], b[M,4] xyxy -> IoU[N,M]."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def match_predictions(iou_gt_pred: np.ndarray, iouv: np.ndarray = IOUV) -> np.ndarray:
    """
    Ultralytics eslestirmesi: her esikte IoU'su en yuksek (gt, tespit) ciftleri
    once alinir, her tespit ve her gt en fazla bir kez kullanilir.
    Not: COCO'dan farkli olarak skor sirasina degil IoU'ya gore acgozlu.
    iou_gt_pred[M,N] -> correct[N, len(iouv)] bool
    """
    n_pred = iou_gt_pred.shape[1]
    correct = np.zeros((n_pred, len(iouv)), dtype=bool)
    for i, thr in enumerate(iouv):
        matches = np.array(np.nonzero(iou_gt_pred >= thr)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_gt_pred[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def image_stats(gt: dict, dets: dict) -> dict:
    """{stem: (tp[N,10], conf[N], n_gt)}. gt'deki her goruntu sayilir."""
    stats = {}
    for stem, gt_boxes in gt.items():
        boxes, scores = dets.get(stem, (np.zeros((0, 4)), np.zeros(0)))
        if len(gt_boxes) == 0 or len(boxes) == 0:
            tp = np.zeros((len(boxes), len(IOUV)), dtype=bool)
        else:
            tp = match_predictions(box_iou(gt_boxes, boxes))
        stats[stem] = (tp, np.asarray(scores, dtype=float), len(gt_boxes))
    return stats


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101 noktali interpolasyon, Ultralytics 8.4.131 ile ayni sentinel degerleri."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def summarize(stats: dict, stems=None) -> dict:
    """
    Tek sinif icin Ultralytics ap_per_class. stems verilirse yalnizca onlar
    (tekrar edebilir, bootstrap icin) kullanilir.
    P ve R, F1'in en yuksek oldugu skor esiginde (Ultralytics tablosu gibi).
    """
    stems = list(stats) if stems is None else list(stems)
    parts = [stats[s] for s in stems]
    n_gt = int(sum(p[2] for p in parts))
    tp = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, len(IOUV)), bool)
    conf = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)

    out = {"n_images": len(stems), "n_gt": n_gt, "n_dets": int(len(conf))}
    if n_gt == 0 or len(conf) == 0:
        out.update(precision=0.0, recall=0.0, f1=0.0, conf_at_max_f1=None,
                   map50=0.0, map50_95=0.0, ap_per_iou=[0.0] * len(IOUV))
        return out

    order = np.argsort(-conf, kind="stable")
    tp, conf = tp[order].astype(float), conf[order]
    tpc = tp.cumsum(0)
    fpc = (1 - tp).cumsum(0)
    recall = tpc / (n_gt + EPS)
    precision = tpc / (tpc + fpc)

    x = np.linspace(0, 1, 1000)
    r_curve = np.interp(-x, -conf, recall[:, 0], left=0)
    p_curve = np.interp(-x, -conf, precision[:, 0], left=1)
    ap = np.array([compute_ap(recall[:, j], precision[:, j]) for j in range(tp.shape[1])])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + EPS)
    i = smooth(f1_curve, 0.1).argmax()
    out.update(
        precision=float(p_curve[i]),
        recall=float(r_curve[i]),
        f1=float(f1_curve[i]),
        conf_at_max_f1=float(x[i]),
        map50=float(ap[0]),
        map50_95=float(ap.mean()),
        ap_per_iou=[float(v) for v in ap],
    )
    return out


def evaluate(gt: dict, dets: dict) -> dict:
    return summarize(image_stats(gt, dets))


def paired_bootstrap(stats_by_method: dict, n_boot: int = 1000, seed: int = 0,
                     metrics=("map50", "map50_95")) -> dict:
    """
    Ayni goruntu orneklemi tum yontemlere uygulanir (eslestirilmis bootstrap),
    boylece yontemler arasi FARKIN guven araligi hesaplanir.
    Dikkat: goruntuler ayni klipte birbirine benzer (bagimsiz degil), bu yuzden
    goruntu bazli bootstrap araligi gercek belirsizligi KUCUK gosterir.
    Klip bazli sonuclarla birlikte okunmali.
    """
    methods = list(stats_by_method)
    stems = sorted(next(iter(stats_by_method.values())))
    for m in methods:
        if sorted(stats_by_method[m]) != stems:
            raise ValueError(f"{m}: goruntu kumesi digerleriyle ayni degil")
    rng = np.random.default_rng(seed)
    samples = {m: {k: [] for k in metrics} for m in methods}
    for _ in range(n_boot):
        idx = rng.integers(0, len(stems), len(stems))
        pick = [stems[i] for i in idx]
        for m in methods:
            res = summarize(stats_by_method[m], pick)
            for k in metrics:
                samples[m][k].append(res[k])
    return {m: {k: np.asarray(v) for k, v in d.items()} for m, d in samples.items()}


def greedy_tp_labels(gt_boxes: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                     iou_thr: float = 0.5) -> np.ndarray:
    """
    Kalibrasyon etiketi: skor sirasiyla (COCO usulu) her tespiti henuz
    eslesmemis en yuksek IoU'lu gercek kutuya baglar. Etiket 1 = "bu tespit
    dogru bir tespit". Bir gercek kutuya ikinci kez baglanan tespit 0 alir,
    cunku mAP de onu yanlis alarm sayar.
    """
    labels = np.zeros(len(boxes), dtype=int)
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return labels
    iou = box_iou(boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for k in np.argsort(-scores, kind="stable"):
        cand = np.where(~used & (iou[k] >= iou_thr))[0]
        if len(cand):
            g = cand[np.argmax(iou[k, cand])]
            used[g] = True
            labels[k] = 1
    return labels
