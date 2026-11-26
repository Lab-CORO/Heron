"""Figures of an evaluation, drawn on the host from eval_data.h5 (needs matplotlib).

Ported unchanged from the former unet_3d/script/evaluate.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heron.evaluation.metrics import error_cdf_from_cm, region_masks, to_classes

_HAS_MATPLOTLIB = True

def plot_confusion_matrix(cls_metrics, output_path):
    if not _HAS_MATPLOTLIB:
        return
    from matplotlib.colors import LogNorm

    cm = cls_metrics["ConfusionMatrix"].astype(np.float64)
    n = cm.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    # Log colour: class 0 (empty space) dominates by orders of magnitude.
    im = ax.imshow(cm, origin="lower", cmap="viridis",
                   norm=LogNorm(vmin=1, vmax=max(cm.max(), 1)))
    ax.plot([0, n - 1], [0, n - 1], "r--", lw=0.8, alpha=0.6)  # perfect-agreement diagonal
    ax.set_xlabel("Predicted class (0–50)")
    ax.set_ylabel("Actual class (0–50)")
    ax.set_title(
        f"Confusion Matrix ({n} classes)  |  "
        f"Exact={cls_metrics['ExactAccuracy']:.3f}  "
        f"±1={cls_metrics['Within1Accuracy']:.3f}  "
        f"QWK={cls_metrics['QuadraticKappa']:.3f}",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="voxel count (log)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_tolerance_curves(region_cm, region_n, output_path, thresh=0.95, kmax=15,
                          global_cm=None, global_n=None, ymin=0.1):
    """Cumulative accuracy vs tolerance: fraction of voxels within +-k classes,
    one curve per region (+ optional global). Marks where each curve first
    reaches `thresh` (95%)."""
    if not _HAS_MATPLOTLIB:
        return
    colors = {"global": "tab:green", "reachable": "tab:blue",
              "graded": "tab:orange", "thin": "tab:red"}
    ks = np.arange(0, kmax + 1)

    # global first (so it draws underneath the region curves), then the regions
    series = []
    if global_cm is not None:
        series.append(("global", global_cm, global_n))
    series += [(r, region_cm[r], region_n[r]) for r in region_cm]

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, cm, n in series:
        cdf = error_cdf_from_cm(cm)
        k95 = int(np.argmax(cdf >= thresh)) if (cdf >= thresh).any() else None
        n0 = int(round(cdf[0] * n)) if n else 0   # exact-correct voxel count
        style = dict(marker="o", ms=3, color=colors.get(r))
        if r == "global":
            style.update(ls="--", lw=1.3, alpha=0.8)   # dashed to set it apart
        lbl = (f"{r}: exact={cdf[0]:.3f} ({n0:,} vox)"
               + (f", {thresh:.0%} at ±{k95}" if k95 is not None else f", <{thresh:.0%}"))
        ax.plot(ks, cdf[:kmax + 1], label=lbl, **style)
        if k95 is not None and k95 <= kmax:
            ax.axvline(k95, color=colors.get(r), ls=":", lw=0.9, alpha=0.6)

    ax.axhline(thresh, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("tolerance  ±k classes   (|pred − label| ≤ k)")
    ax.set_ylabel("fraction of voxels within tolerance")
    ax.set_title("Cumulative accuracy vs tolerance, by region")
    ax.set_xlim(0, kmax); ax.set_ylim(ymin, 1.01)
    ax.set_xticks(np.arange(0, kmax + 1, 1))
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_histogram(hist_counts, hist_edges, mean_err, median_err, output_path):
    if not _HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    widths = np.diff(hist_edges)
    ax.bar(hist_edges[:-1], hist_counts, width=widths, align="edge",
           color="steelblue", edgecolor="none")
    ax.axvline(mean_err,   color="red",    linestyle="--", label=f"Mean  {mean_err:.4f}")
    ax.axvline(median_err, color="orange", linestyle="--", label=f"Median {median_err:.4f}")
    ax.set_xlabel("|pred − label|")
    ax.set_ylabel("Density")
    ax.set_title("Absolute Error Distribution (all val voxels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_region_slices(samples, output_dir):
    """For each (pred, label, raw) sample, save a regions_N.png showing:
      col 0: raw input (binary obstacle occupancy, 0/1)
      col 1: label where reachable (class>=1), black outside
      col 2: label where graded (class 1-49), black outside
      col 3: label where thin (top-1% |grad(label)|), black outside

    Uses the X-axis slice with the most thin-region voxels (singularity plane).
    """
    if not _HAS_MATPLOTLIB:
        return
    col_titles = ["raw input\n(obstacles)", "reachable\n(class ≥ 1)",
                  "graded\n(class 1–49)", "thin\n(top 1% |∇label|)"]
    for n, (pred, label, raw) in enumerate(samples):
        label_cls = to_classes(label)
        masks = region_masks(label_cls, label)

        # pick X slice with most thin-region voxels (singularity plane)
        thin_per_x = masks["thin"].sum(axis=(1, 2))
        xc = int(np.argmax(thin_per_x)) if thin_per_x.max() > 0 else label.shape[0] // 2

        raw_sl   = np.rot90(raw[xc, :, :])
        label_sl = np.rot90(label[xc, :, :])
        mask_sls = {r: np.rot90(masks[r][xc, :, :]) for r in ["reachable", "graded", "thin"]}

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs_flat = axs.flatten()   # [top-left, top-right, bottom-left, bottom-right]

        # top-left: raw input — binary occupancy (0 or 1), hot colormap
        axs_flat[0].imshow(raw_sl, cmap="hot", vmin=0, vmax=1)
        axs_flat[0].set_title(col_titles[0]); axs_flat[0].axis("off")
        n_obs = int((raw_sl > 0).sum())
        axs_flat[0].set_xlabel(f"{n_obs} occupied voxels", fontsize=8)

        # top-right + bottom row: label value inside region, white outside
        cmap_region = plt.cm.gnuplot.copy(); cmap_region.set_bad("white")
        for ax, r, title in zip(axs_flat[1:], ["reachable", "graded", "thin"], col_titles[1:]):
            m = mask_sls[r]
            img = np.where(m, label_sl.astype(np.float32), np.nan)
            im = ax.imshow(img, cmap=cmap_region, vmin=0, vmax=1)
            ax.set_title(title); ax.axis("off")
            ax.set_xlabel(f"{int(m.sum())} voxels in slice", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Sample {n}  —  X slice {xc}  (singularity plane)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"regions_{n}.png"), dpi=150)
        plt.close()
