"""eval_data.h5: everything needed to redo the report and figures without the model."""
import h5py

from heron.evaluation.metrics import REGION_NAMES

EVAL_DATA_FILENAME = "eval_data.h5"


def save_eval_data(path, global_cm, region_cm, n_files, global_reg,
                    hist_counts, hist_edges, region_samples):
    """Persist everything generate_outputs() needs, so plots/report can be redone
    later via --plot_only without rerunning the model over the val set."""
    with h5py.File(path, "w") as f:
        f.attrs["n_files"] = n_files
        for k, v in global_reg.items():
            f.attrs[k] = v

        f.create_dataset("global_cm", data=global_cm, compression="gzip")
        cm_grp = f.create_group("region_cm")
        for r in REGION_NAMES:
            cm_grp.create_dataset(r, data=region_cm[r], compression="gzip")

        f.create_dataset("hist_counts", data=hist_counts, compression="gzip")
        f.create_dataset("hist_edges", data=hist_edges, compression="gzip")

        samples_grp = f.create_group("region_samples")
        for i, (pred, label, raw) in enumerate(region_samples):
            sg = samples_grp.create_group(str(i))
            sg.create_dataset("pred", data=pred, compression="gzip")
            sg.create_dataset("label", data=label, compression="gzip")
            sg.create_dataset("raw", data=raw, compression="gzip")


def load_eval_data(path):
    with h5py.File(path, "r") as f:
        global_cm = f["global_cm"][:]
        region_cm = {r: f["region_cm"][r][:] for r in REGION_NAMES}
        n_files = int(f.attrs["n_files"])
        global_reg = {"MAE": float(f.attrs["MAE"]), "MedAE": float(f.attrs["MedAE"]),
                      "RMSE": float(f.attrs["RMSE"]), "Pearson_r": float(f.attrs["Pearson_r"])}
        hist_counts = f["hist_counts"][:]
        hist_edges = f["hist_edges"][:]

        region_samples = []
        sg = f["region_samples"]
        for k in sorted(sg.keys(), key=int):
            grp = sg[k]
            region_samples.append((grp["pred"][:], grp["label"][:], grp["raw"][:]))

    return {
        "global_cm": global_cm, "region_cm": region_cm, "n_files": n_files,
        "global_reg": global_reg, "hist_counts": hist_counts, "hist_edges": hist_edges,
        "region_samples": region_samples,
    }
