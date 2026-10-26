"""report.txt: the text report of one evaluated part (no matplotlib needed)."""

_REGION_DESC = {
    "reachable": "class >= 1 (non-empty)",
    "graded":    "1..49 (transition shell)",
    "thin":      "top-1% |grad(label)| (edges/singularity lines)",
}


def save_report(reg, cls, output_path, n_files, n_voxels, region_cls=None, region_n=None):
    with open(output_path, "w") as f:
        f.write("=== CapaciNet Evaluation Report ===\n\n")
        f.write(f"Files     : {n_files}\n")
        f.write(f"Voxels    : {n_voxels:,}\n\n")
        f.write("-- Regression (continuous) --\n")
        f.write(f"MAE       : {reg['MAE']:.6f}\n")
        f.write(f"MedAE     : {reg['MedAE']:.6f}\n")
        f.write(f"RMSE      : {reg['RMSE']:.6f}\n")
        f.write(f"Pearson r : {reg['Pearson_r']:.6f}\n\n")
        f.write("-- Classification (51 classes, 0-50; class = round(value*50)) --\n")
        f.write(f"Exact accuracy   : {cls['ExactAccuracy']:.4f}\n")
        f.write(f"Within-+-1 acc.  : {cls['Within1Accuracy']:.4f}\n")
        f.write(f"Within-+-2 acc.  : {cls['Within2Accuracy']:.4f}\n")
        f.write(f"Within-+-10 acc. : {cls['Within10Accuracy']:.4f}\n")
        f.write(f"Within-+-20 acc. : {cls['Within20Accuracy']:.4f}\n")
        f.write(f"Mean class error : {cls['MeanClassError']:.4f}  (orientation steps)\n")
        f.write(f"RMSE (class)     : {cls['RMSEClass']:.4f}\n")
        f.write(f"Quadratic kappa  : {cls['QuadraticKappa']:.4f}\n")
        if region_cls:
            f.write("\n-- Per-region classification (background excluded) --\n")
            f.write(f"{'region':<10} {'voxels':>12}  {'exact':>6} {'+-1':>6} {'+-2':>6} {'+-10':>6} {'+-20':>6} "
                    f"{'meanErr':>8} {'rmse':>6} {'QWK':>6}\n")
            for r, m in region_cls.items():
                f.write(f"{r:<10} {region_n[r]:>12,}  "
                        f"{m['ExactAccuracy']:>6.3f} {m['Within1Accuracy']:>6.3f} "
                        f"{m['Within2Accuracy']:>6.3f} {m['Within10Accuracy']:>6.3f} "
                        f"{m['Within20Accuracy']:>6.3f} {m['MeanClassError']:>8.3f} "
                        f"{m['RMSEClass']:>6.2f} {m['QuadraticKappa']:>6.3f}\n")
            for r in region_cls:
                f.write(f"  {r}: {_REGION_DESC[r]}\n")


def summarize(global_cm, region_cm, global_reg) -> dict:
    """Scalar metrics of one part, from its accumulated confusion matrices."""
    from heron.evaluation.metrics import REGION_NAMES, merge_class_metrics

    def scalars(m):
        return {k: v for k, v in m.items() if k not in ("ConfusionMatrix", "MedianClassError")}

    global_n = int(global_cm.sum())
    region_n = {r: int(region_cm[r].sum()) for r in REGION_NAMES}
    return {
        "voxels": global_n,
        "regression": {k: float(v) for k, v in global_reg.items()},
        "classification": scalars(merge_class_metrics(global_cm, n_voxels=global_n)),
        "regions": {r: scalars(merge_class_metrics(region_cm[r], n_voxels=region_n[r])) for r in REGION_NAMES},
    }


def write_report(output_dir, global_cm, region_cm, n_files, global_reg) -> None:
    """report.txt of one part, in the historical layout."""
    import os
    from heron.evaluation.metrics import REGION_NAMES, merge_class_metrics

    global_n = int(global_cm.sum())
    region_n = {r: int(region_cm[r].sum()) for r in REGION_NAMES}
    save_report(global_reg, merge_class_metrics(global_cm, n_voxels=global_n),
                os.path.join(output_dir, "report.txt"), n_files=n_files, n_voxels=global_n,
                region_cls={r: merge_class_metrics(region_cm[r], n_voxels=region_n[r]) for r in REGION_NAMES},
                region_n=region_n)
