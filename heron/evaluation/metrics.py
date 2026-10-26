"""Metrics of a predicted Reachability map against the generated one.

Ported unchanged from unet_3d/script/evaluate.py so reports stay comparable.
The reachability index is a fraction of 50 (n reachable orientations -> n/50), so
classification treats each voxel as one of 51 ordinal classes via
class = round(value * 50), applied to BOTH prediction and ground truth.
"""
import numpy as np
from scipy.stats import pearsonr

N_CLASSES = 51  # reachability index is a fraction of 50 -> classes 0..50
REGION_NAMES = ["reachable", "graded", "thin"]


def to_classes(arr):
    """Quantize a [0,1] reachability fraction into integer classes 0..50.

    Each voxel's reachability is n/50 (n reachable orientations), so the natural
    class label is round(value * 50). Predictions are post-sigmoid (model in eval
    mode), but clip guards any tiny over/undershoot.
    """
    return np.clip(np.rint(arr * 50.0), 0, 50).astype(np.int16)


def gradmag3d(a):
    """Per-voxel gradient magnitude of a 3D array (for locating sharp/thin edges)."""
    gx, gy, gz = np.gradient(a.astype(np.float32))
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def region_masks(label_cls, label_cont, grad_pct=99.0):
    """Boolean masks (same shape as the volume) isolating where the model's job is
    actually hard. Aggregates dominated by empty background hide these.

      reachable : class >= 1            (any non-empty voxel)
      graded    : 1 <= class <= 49      (transition shell: not empty, not saturated)
      thin      : top-1% |grad(label)|  (sharp boundaries / singularity lines)
    """
    reachable = label_cls >= 1
    graded    = (label_cls >= 1) & (label_cls <= 49)
    gl = gradmag3d(label_cont)
    if reachable.any():
        thr = np.quantile(gl, grad_pct / 100.0)
        thin = gl >= thr
    else:
        thin = np.zeros_like(reachable)
    return {"reachable": reachable, "graded": graded, "thin": thin}


def regression_metrics(preds, labels):
    # float64 throughout: at N=4B, float32 accumulation loses precision and saturates
    # pearsonr to 1.000000 (terms ~1e-10 fall below float32 eps once sum > 1e-3).
    p = preds.astype(np.float64)
    l = labels.astype(np.float64)
    diff = p - l
    abs_diff = np.abs(diff)
    mae   = float(np.mean(abs_diff))
    medae = float(np.median(abs_diff))  # stored so the error histogram can be replotted from cached bins alone
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    r, _ = pearsonr(p, l)
    return {"MAE": mae, "MedAE": medae, "RMSE": rmse, "Pearson_r": float(r)}


def confusion_matrix(pred_cls, label_cls, n_classes=N_CLASSES):
    """Dense (n_classes x n_classes) confusion matrix (rows=actual, cols=pred)."""
    idx = label_cls.astype(np.int64) * n_classes + pred_cls.astype(np.int64)
    cm = np.bincount(idx, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes).astype(np.int64)


def quadratic_weighted_kappa(cm):
    """Cohen's quadratic-weighted kappa from a confusion matrix (ordinal agreement)."""
    n = cm.shape[0]
    total = cm.sum()
    if total == 0:
        return 0.0
    O = cm.astype(np.float64) / total                 # observed joint distribution
    act = O.sum(axis=1)                                # actual marginal
    prd = O.sum(axis=0)                                # predicted marginal
    E = np.outer(act, prd)                             # expected under independence
    i = np.arange(n)
    W = (i[:, None] - i[None, :]) ** 2 / float((n - 1) ** 2)
    denom = float((W * E).sum())
    if denom == 0:
        return 1.0
    return float(1.0 - (W * O).sum() / denom)


def multiclass_metrics(pred_cls, label_cls, n_classes=N_CLASSES):
    """Ordinal 51-class metrics measured on the integer-class arrays."""
    diff = np.abs(pred_cls.astype(np.int32) - label_cls.astype(np.int32))
    n = diff.size

    exact    = float(np.mean(diff == 0))
    within1  = float(np.mean(diff <= 1))
    within2  = float(np.mean(diff <= 2))
    within10 = float(np.mean(diff <= 10))
    within20 = float(np.mean(diff <= 20))
    mace    = float(np.mean(diff))          # mean absolute class error (orientation steps)
    medce   = float(np.median(diff))
    rmsece  = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))

    cm = confusion_matrix(pred_cls, label_cls, n_classes)
    qwk = quadratic_weighted_kappa(cm)

    return {
        "ExactAccuracy": exact,
        "Within1Accuracy": within1,
        "Within2Accuracy": within2,
        "Within10Accuracy": within10,
        "Within20Accuracy": within20,
        "MeanClassError": mace,
        "MedianClassError": medce,
        "RMSEClass": rmsece,
        "QuadraticKappa": qwk,
        "N": int(n),
        "ConfusionMatrix": cm,
    }


def merge_class_metrics(cm, n_voxels):
    """Recompute scalar metrics from an accumulated confusion matrix.

    Lets the global aggregate be derived by summing per-file confusion matrices
    instead of concatenating every voxel.
    """
    n = cm.shape[0]
    i = np.arange(n)
    diff = np.abs(i[:, None] - i[None, :])             # (actual, pred) class distance
    counts = cm.astype(np.float64)
    total = counts.sum()
    if total == 0:
        total = 1.0

    exact    = float(np.trace(counts) / total)
    within1  = float(counts[diff <= 1].sum() / total)
    within2  = float(counts[diff <= 2].sum() / total)
    within10 = float(counts[diff <= 10].sum() / total)
    within20 = float(counts[diff <= 20].sum() / total)
    mace    = float((counts * diff).sum() / total)
    rmsece  = float(np.sqrt((counts * diff.astype(np.float64) ** 2).sum() / total))

    return {
        "ExactAccuracy": exact,
        "Within1Accuracy": within1,
        "Within2Accuracy": within2,
        "Within10Accuracy": within10,
        "Within20Accuracy": within20,
        "MeanClassError": mace,
        "MedianClassError": float("nan"),  # not recoverable from the matrix
        "RMSEClass": rmsece,
        "QuadraticKappa": quadratic_weighted_kappa(cm),
        "N": int(n_voxels),
        "ConfusionMatrix": cm,
    }


def error_cdf_from_cm(cm):
    """From a confusion matrix, return cdf[k] = fraction of voxels with
    |pred - label| <= k  (k = 0..n_classes-1). cdf[0] is the exact accuracy."""
    n = cm.shape[0]
    i = np.arange(n)
    dist = np.abs(i[:, None] - i[None, :]).ravel()
    counts = np.bincount(dist, weights=cm.ravel().astype(np.float64), minlength=n)
    total = counts.sum()
    return np.cumsum(counts) / (total if total > 0 else 1.0)


def compute_error_histogram(preds, labels, bins=100):
    """Bin |pred - label| once, at predict time, so the histogram plot can later be
    redrawn from these small arrays alone instead of the (potentially huge) raw
    per-voxel preds/labels."""
    errors = np.abs(preds.astype(np.float64) - labels.astype(np.float64))
    counts, edges = np.histogram(errors, bins=bins, density=True)
    return counts, edges
