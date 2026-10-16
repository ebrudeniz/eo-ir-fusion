"""
Karar seviyesi EO/IR fuzyonu
============================

Akis (her goruntu icin):
  1) Kalibrasyon: her sensorun skoru, val'de ogrenilen Platt olcekleme ile
     "bu kutu dogru tespit" olasiligina cevrilir. Neden: iki dedektorun 0,7'si
     ayni sey demek degil; olasiliga cevirmeden toplamak/kiyaslamak anlamsiz.
     Tek sensorun mAP'ini degistirmez (monoton donusum siralamayi bozmaz),
     yalnizca iki sensor birlesirken etkisi olur.
  2) Eslestirme: EO ve IR kutulari IoU >= esik olan ciftler arasindan,
     guclu kutular once (cift icindeki yuksek olasilik, esitlikte IoU)
     acgozlu bire bir eslenir. Her kume en fazla 1 EO + 1 IR kutusu tasir.
     Eslesmeyen kutu tek sensorlu kume olarak kalir.
  3) Skor kurali ve kutu kurali kumeyi tek tespite cevirir.
  4) Istege bagli son NMS (final_nms): bire bir eslestirme ayni sensordeki kopya
     kutulari (NMS 0,7'den kurtulmus, IoU < 0,7) ayri tespit olarak birakir ve
     bunlar yanlis alarm sayilir. WBF bunlari ayni kumeye katarak yutar. Adil
     olsun diye ayni NMS baseline'lara da uygulanir (single_sensor_nms), yoksa
     kazancin fuzyondan mi NMS ayarindan mi geldigi ayrilamaz.
  5) Goruntu basina en fazla max_det (300) tespit: baseline'lar da Ultralytics'te
     300 ile sinirli, fuzyon 600'e kadar kutu birakirsa karsilastirma adil olmaz.

Skor kurallari (p_eo, p_ir kalibre olasilik; eksik sensor = 0):
  max           max(p_eo, p_ir)                    birlesim (union) davranisi
  mean          (p_eo + p_ir) / 2                  tek sensorlu kutuyu cezalandirir
  noisy_or      1 - (1 - p_eo)(1 - p_ir)           hatalarin bagimsiz oldugu varsayimi
  intersection  yalnizca eslesen kumeler, mean     kesisim (iki sensor de gormeli)
  learned       lojistik regresyon (asagida)

Kutu kurallari:
  ir        IR kutusu varsa o, yoksa EO
  score     olasiligi yuksek olan sensorun kutusu
  weighted  olasilikla agirlikli ortalama (WBF'in kutu birlestirmesi)
"""

import numpy as np

from .metrics import box_iou, greedy_tp_labels

SCORE_RULES = ("max", "mean", "noisy_or", "intersection", "learned")
BOX_RULES = ("ir", "score", "weighted")
LOGIT_CLIP = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 100) -> np.ndarray:
    """
    Newton (IRLS) ile L2'li lojistik regresyon. X'e kesisim sutunu eklenir,
    kesisim cezalandirilmaz. Donen w: [kesisim, w_1..w_d].
    sklearn bagimliligi eklememek icin elle; birkac ozellik icin yeterli.
    """
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    y = np.asarray(y, dtype=float)
    if y.min() == y.max():
        raise ValueError("Lojistik regresyon icin iki sinif da gerekli")
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2 * len(y))
    reg[0] = 0.0
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + reg * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logistic_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    return sigmoid(X @ w)


class Calibrator:
    """Platt olcekleme: p = sigmoid(a * logit(s) + b). method='none' kimlik."""

    def __init__(self, method: str = "platt", a: float = 1.0, b: float = 0.0):
        if method not in ("platt", "none"):
            raise ValueError(method)
        self.method, self.a, self.b = method, float(a), float(b)

    def fit(self, scores, labels):
        if self.method == "platt":
            w = logistic_fit(logit(scores)[:, None], labels)
            self.b, self.a = float(w[0]), float(w[1])
        return self

    def __call__(self, scores):
        scores = np.asarray(scores, dtype=float)
        if self.method == "none":
            return scores
        return sigmoid(self.a * logit(scores) + self.b)

    def to_dict(self):
        return {"method": self.method, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d):
        return cls(d["method"], d["a"], d["b"])


def calibration_data(gt: dict, dets: dict, stems, iou_thr: float = 0.5):
    """Secilen goruntulerdeki tum tespitler icin (skor, dogru mu) dizileri."""
    s, y = [], []
    for stem in stems:
        boxes, scores = dets[stem]
        s.append(scores)
        y.append(greedy_tp_labels(gt[stem], boxes, scores, iou_thr))
    return np.concatenate(s), np.concatenate(y)


def expected_calibration_error(p, y, n_bins: int = 15):
    """ECE ve guvenilirlik diyagrami icin kutu istatistikleri."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf, acc = p[m].mean(), y[m].mean()
            ece += m.mean() * abs(conf - acc)
            bins.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "mean_p": float(conf), "frac_pos": float(acc)})
    return float(ece), bins


def match_pairs(b_eo, p_eo, b_ir, p_ir, iou_thr: float):
    """
    Bire bir acgozlu eslestirme. Donus: (eo_idx, ir_idx, pair_iou) dizileri;
    eksik taraf -1, tek sensorlu kumede pair_iou 0.
    """
    n_eo, n_ir = len(b_eo), len(b_ir)
    eo_idx, ir_idx, pair_iou = [], [], []
    used_eo = np.zeros(n_eo, dtype=bool)
    used_ir = np.zeros(n_ir, dtype=bool)
    if n_eo and n_ir:
        iou = box_iou(b_eo, b_ir)
        ie, ji = np.nonzero(iou >= iou_thr)
        if len(ie):
            strength = np.maximum(p_eo[ie], p_ir[ji])
            order = np.lexsort((-iou[ie, ji], -strength))
            for k in order:
                e, j = ie[k], ji[k]
                if not used_eo[e] and not used_ir[j]:
                    used_eo[e] = used_ir[j] = True
                    eo_idx.append(e)
                    ir_idx.append(j)
                    pair_iou.append(iou[e, j])
    rest_eo = np.where(~used_eo)[0]
    rest_ir = np.where(~used_ir)[0]
    eo_idx = np.concatenate([np.asarray(eo_idx, int), rest_eo, -np.ones(len(rest_ir), int)])
    ir_idx = np.concatenate([np.asarray(ir_idx, int), -np.ones(len(rest_eo), int), rest_ir])
    pair_iou = np.concatenate([np.asarray(pair_iou, float), np.zeros(len(rest_eo) + len(rest_ir))])
    return eo_idx, ir_idx, pair_iou


def _gather(arr, idx, fill):
    out = np.full((len(idx),) + arr.shape[1:], fill, dtype=float)
    ok = idx >= 0
    if ok.any():
        out[ok] = arr[idx[ok]]
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Klasik acgozlu NMS; tutulan indeksleri skor sirasiyla dondurur."""
    order = np.argsort(-scores, kind="stable")
    if len(order) == 0:
        return order
    iou = box_iou(boxes[order], boxes[order])
    keep = np.ones(len(order), dtype=bool)
    for a in range(len(order)):
        if keep[a]:
            keep[a + 1:] &= iou[a, a + 1:] < iou_thr
    return order[keep]


def single_sensor_nms(dets: dict, iou_thr, max_det: int = 300) -> dict:
    """Baseline'in ayni son NMS ile hali (iou_thr=None ise oldugu gibi)."""
    if iou_thr is None:
        return dets
    out = {}
    for stem, (b, s) in dets.items():
        k = nms(b, s, iou_thr)[:max_det]
        out[stem] = (b[k], s[k])
    return out


def build_clusters(eo, ir, cal_eo, cal_ir, match_iou):
    """Bir goruntu icin kume tablosu (sozluk). eo/ir: (boxes, scores)."""
    b_eo, s_eo = eo
    b_ir, s_ir = ir
    p_eo_all, p_ir_all = cal_eo(s_eo), cal_ir(s_ir)
    e, i, piou = match_pairs(b_eo, p_eo_all, b_ir, p_ir_all, match_iou)
    return {
        "has_eo": e >= 0, "has_ir": i >= 0, "iou": piou,
        "p_eo": _gather(p_eo_all, e, 0.0), "p_ir": _gather(p_ir_all, i, 0.0),
        "s_eo": _gather(s_eo, e, 0.0), "s_ir": _gather(s_ir, i, 0.0),
        "b_eo": _gather(b_eo.reshape(-1, 4), e, np.nan), "b_ir": _gather(b_ir.reshape(-1, 4), i, np.nan),
    }


def cluster_boxes(c, box_rule: str) -> np.ndarray:
    both = c["has_eo"] & c["has_ir"]
    only_eo = c["has_eo"] & ~c["has_ir"]
    out = np.where(only_eo[:, None], c["b_eo"], c["b_ir"])  # tek sensorlu kume kendi kutusunu alir
    if box_rule == "ir":
        pass
    elif box_rule == "score":
        eo_wins = both & (c["p_eo"] > c["p_ir"])
        out[eo_wins] = c["b_eo"][eo_wins]
    elif box_rule == "weighted":
        we, wi = c["p_eo"][both, None], c["p_ir"][both, None]
        out[both] = (we * c["b_eo"][both] + wi * c["b_ir"][both]) / (we + wi + 1e-12)
    else:
        raise ValueError(box_rule)
    return out


def learned_features(c, boxes) -> np.ndarray:
    """
    Ogrenilmis fuzyon ozellikleri. Eksik sensorun skoru 0 yapmak yerine
    has_* gostergeleriyle ayrica veriliyor: "sensor gormedi" bilgisi kendi
    basina sinyal, skoru 0 yapmak bunu "cok dusuk skor"la karistirir.
    """
    he, hi = c["has_eo"].astype(float), c["has_ir"].astype(float)
    area = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1.0, None)
    return np.column_stack([
        logit(c["s_eo"]) * he, logit(c["s_ir"]) * hi, he, hi, c["iou"], np.log(area),
    ])


class FusionModel:
    def __init__(self, match_iou=0.5, score_rule="max", box_rule="ir", calibration="platt",
                 final_nms=None, max_det=300):
        if score_rule not in SCORE_RULES or box_rule not in BOX_RULES:
            raise ValueError((score_rule, box_rule))
        self.cfg = {"match_iou": float(match_iou), "score_rule": score_rule, "box_rule": box_rule,
                    "calibration": calibration,
                    "final_nms": None if final_nms is None else float(final_nms), "max_det": int(max_det)}
        self.cal_eo = Calibrator(calibration)
        self.cal_ir = Calibrator(calibration)
        self.learned = None  # {"w":..., "mu":..., "sd":...}

    # ---- egitim (yalnizca val klipleri uzerinde cagrilir) -------------------
    def fit(self, gt, eo, ir, stems, calibrators=None, clusters=None):
        """
        calibrators=(cal_eo, cal_ir) verilirse kalibrasyon yeniden ogrenilmez
        (capraz dogrulamada ayni fold'daki tum ayarlar ayni kalibrasyonu paylasir).
        clusters={stem: build_clusters(...)} verilirse eslestirme tekrarlanmaz;
        bu kumeler AYNI kalibrasyon ve AYNI match_iou ile kurulmus olmali.
        """
        stems = list(stems)
        if calibrators is not None:
            self.cal_eo, self.cal_ir = calibrators
        else:
            self.cal_eo.fit(*calibration_data(gt, eo, stems))
            self.cal_ir.fit(*calibration_data(gt, ir, stems))
        if self.cfg["score_rule"] == "learned":
            X, y = [], []
            for stem in stems:
                c = clusters[stem] if clusters is not None else self.clusters(eo[stem], ir[stem])
                boxes = cluster_boxes(c, self.cfg["box_rule"])
                order_score = np.maximum(c["p_eo"], c["p_ir"])
                X.append(learned_features(c, boxes))
                y.append(greedy_tp_labels(gt[stem], boxes, order_score))
            X, y = np.concatenate(X), np.concatenate(y)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            self.learned = {"w": logistic_fit((X - mu) / sd, y), "mu": mu, "sd": sd}
        return self

    # ---- uygulama ---------------------------------------------------------
    def clusters(self, eo, ir):
        return build_clusters(eo, ir, self.cal_eo, self.cal_ir, self.cfg["match_iou"])

    def fuse_image(self, eo, ir, c=None):
        cfg = self.cfg
        c = self.clusters(eo, ir) if c is None else c
        boxes = cluster_boxes(c, cfg["box_rule"])
        pe, pi = c["p_eo"], c["p_ir"]
        rule = cfg["score_rule"]
        keep = np.ones(len(boxes), dtype=bool)
        if rule == "max":
            score = np.maximum(pe, pi)
        elif rule == "mean":
            score = (pe + pi) / 2
        elif rule == "noisy_or":
            score = 1 - (1 - pe) * (1 - pi)
        elif rule == "intersection":
            score = (pe + pi) / 2
            keep = c["has_eo"] & c["has_ir"]
        elif rule == "learned":
            L = self.learned
            score = logistic_predict(L["w"], (learned_features(c, boxes) - L["mu"]) / L["sd"])
        boxes, score = boxes[keep], score[keep]
        if cfg["final_nms"] is not None:
            top = nms(boxes, score, cfg["final_nms"])[: cfg["max_det"]]
        else:
            top = np.argsort(-score, kind="stable")[: cfg["max_det"]]
        return boxes[top], score[top]

    def fuse(self, eo: dict, ir: dict, stems=None, clusters=None) -> dict:
        stems = sorted(eo) if stems is None else stems
        if clusters is None:
            return {s: self.fuse_image(eo[s], ir[s]) for s in stems}
        return {s: self.fuse_image(None, None, clusters[s]) for s in stems}

    # ---- kayit ------------------------------------------------------------
    def to_dict(self):
        d = {"config": self.cfg, "cal_eo": self.cal_eo.to_dict(), "cal_ir": self.cal_ir.to_dict()}
        if self.learned is not None:
            d["learned"] = {k: np.asarray(v).tolist() for k, v in self.learned.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls(**d["config"])
        m.cal_eo = Calibrator.from_dict(d["cal_eo"])
        m.cal_ir = Calibrator.from_dict(d["cal_ir"])
        if "learned" in d:
            m.learned = {k: np.asarray(v, dtype=float) for k, v in d["learned"].items()}
        return m


def wbf_reference(eo: dict, ir: dict, image_wh, iou_thr=0.55, skip_box_thr=0.0, max_det=300):
    """
    Literatur referansi: ensemble_boxes paketinin Weighted Boxes Fusion'i
    (Solovyev ve ark. 2021), ham skorlarla ve esit agirlikla, kendi
    uygulamamizdan bagimsiz olsun diye paket kullaniliyor.
    """
    import warnings

    from ensemble_boxes import weighted_boxes_fusion

    W, H = image_wh
    scale = np.array([W, H, W, H], dtype=float)
    out = {}
    for stem in sorted(eo):
        bl, sl, ll = [], [], []
        for b, s in (eo[stem], ir[stem]):
            bl.append(np.clip(b / scale, 0, 1).tolist())
            sl.append(s.tolist())
            ll.append([0] * len(s))
        if not any(len(s) for s in sl):
            out[stem] = (np.zeros((0, 4)), np.zeros(0))
            continue
        with warnings.catch_warnings():  # kenara kirpilip alani 0 kalan kutular icin uyari
            warnings.simplefilter("ignore", UserWarning)
            boxes, scores, _ = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        top = np.argsort(-scores, kind="stable")[:max_det]
        out[stem] = (boxes[top] * scale, scores[top])
    return out
