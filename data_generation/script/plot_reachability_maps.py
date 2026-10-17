"""
plot_reachability_maps.py — compare the reachability maps of several robots.

Renders a single matplotlib figure with one panel per robot, showing a slice of each
robot's master reachability map along a chosen axis (--axis x|y|z, default z). The
continuous reachability index (0..1) is shown as a heatmap (non-reachable voxels in
white), all panels share a common cropped extent (so physical scales are comparable)
and the axes are in meters (origin + index * voxel_size).

Input files have structure group/0/reachability_map (shape 200^3, float64) with
attributes origine_{x,y,z} and voxel_size.

Examples:
    python plot_reachability_maps.py --axis z --index 100
    python plot_reachability_maps.py --axis x --index 100 --output reach_x.png
"""
import argparse
import os

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless (cluster / no display)
import matplotlib.pyplot as plt


# Default robots -> file (relative to the project root, 3 levels above this script).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_FILES = [
    ("Doosan m1013", os.path.join(PROJECT_ROOT, "data_002/master_ik_data0.02.h5")),
    ("UR5e",         os.path.join(PROJECT_ROOT, "data_ur5e/master_ik_data0.02.h5")),
    ("Jaco7",        os.path.join(PROJECT_ROOT, "data_jaco7/master_ik_data0.02.h5")),
    ("iiwa",         os.path.join(PROJECT_ROOT, "data_iiwa/master_ik_data0.02.h5")),
]


AXES = {"x": 0, "y": 1, "z": 2}
AXIS_NAMES = ("x", "y", "z")


def load_slice(filepath, axis_idx, index):
    """Return (slice_2d, attrs) for a slice of reachability_map along `axis_idx`.

    The returned 2D array is indexed [h, v], where (h, v) are the two non-sliced
    axes in ascending order (axis x->(y,z), y->(x,z), z->(x,y)).
    """
    try:
        f = h5py.File(filepath, "r", locking=False)
    except TypeError:
        f = h5py.File(filepath, "r")
    with f:
        rm = f["group/0/reachability_map"]
        if not (0 <= index < rm.shape[axis_idx]):
            raise ValueError(
                f"index={index} out of range for {filepath} "
                f"(axis '{AXIS_NAMES[axis_idx]}' dim = {rm.shape[axis_idx]})")
        if axis_idx == 0:
            sl = rm[index, :, :]
        elif axis_idx == 1:
            sl = rm[:, index, :]
        else:
            sl = rm[:, :, index]
        attrs = dict(rm.attrs)
    return sl.astype(np.float64), attrs


def reachable_bbox(slices):
    """Union bounding box (x0, x1, y0, y1) of reachable cells across all slices."""
    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    for sl in slices:
        xs, ys = np.where(sl > 0)
        if xs.size:
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
            y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    if not np.isfinite(x0):
        raise RuntimeError("no reachable cells found in any slice at this layer")
    return int(x0), int(x1), int(y0), int(y1)


def main():
    parser = argparse.ArgumentParser(description="Plot robot reachability map slices.")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z",
                        help="Axis to slice along (default: z). z->plane(x,y), y->plane(x,z), x->plane(y,z)")
    parser.add_argument("--index", type=int, default=100, help="Slice index along --axis (default: 100)")
    parser.add_argument("--margin", type=int, default=4, help="Voxel margin around the common bbox (default: 4)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap (default: viridis)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: reachability_<axis><index>.png)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    parser.add_argument(
        "--file", action="append", metavar="NAME=PATH", default=None,
        help="Override a robot file as 'Name=path' (repeatable). Defaults to the 4 robots.",
    )
    args = parser.parse_args()

    robots = ([(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.file]
              if args.file else DEFAULT_FILES)
    axis_idx = AXES[args.axis]
    # The two plotted (horizontal, vertical) axes are the non-sliced ones, ascending.
    h_idx, v_idx = [a for a in (0, 1, 2) if a != axis_idx]
    h_name, v_name = AXIS_NAMES[h_idx], AXIS_NAMES[v_idx]
    output = args.output or f"reachability_{args.axis}{args.index}.png"

    names = [n for n, _ in robots]
    slices, attrs_list = [], []
    for name, path in robots:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name}: file not found: {path}")
        sl, attrs = load_slice(path, axis_idx, args.index)
        slices.append(sl)
        attrs_list.append(attrs)

    # Common cropped extent (with margin), in index space (slices are indexed [h, v]).
    h0, h1, v0, v1 = reachable_bbox(slices)
    nh, nv = slices[0].shape
    h0 = max(0, h0 - args.margin); h1 = min(nh - 1, h1 + args.margin)
    v0 = max(0, v0 - args.margin); v1 = min(nv - 1, v1 + args.margin)

    # Physical extent in meters (use the first robot's geometry; grids are identical).
    a = attrs_list[0]
    origins = [float(a.get("origine_x", 0.0)), float(a.get("origine_y", 0.0)), float(a.get("origine_z", 0.0))]
    vs = float(a.get("voxel_size", 0.02))
    oh, ov = origins[h_idx], origins[v_idx]
    slice_m = origins[axis_idx] + args.index * vs
    extent = [oh + h0 * vs, oh + (h1 + 1) * vs, ov + v0 * vs, ov + (v1 + 1) * vs]

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad("white")  # non-reachable (masked) voxels shown white

    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.2),
                             constrained_layout=True, squeeze=False)
    im = None
    for ax, name, sl in zip(axes.ravel(), names, slices):
        crop = sl[h0:h1 + 1, v0:v1 + 1]
        masked = np.ma.masked_where(crop <= 0, crop)
        # transpose so the H axis is horizontal and V vertical; origin='lower' for +V up
        im = ax.imshow(masked.T, origin="lower", extent=extent, cmap=cmap,
                       vmin=0.0, vmax=1.0, interpolation="nearest")
        n_reach = int((sl > 0).sum())
        ax.set_title(f"{name}  ({n_reach} reachable voxels)")
        ax.set_xlabel(f"{h_name} (m)")
        ax.set_ylabel(f"{v_name} (m)")
        ax.set_aspect("equal")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="reachability index")
    fig.suptitle(
        f"Reachability map — {args.axis.upper()} layer {args.index} "
        f"({args.axis} = {slice_m:.2f} m)", fontsize=15)

    fig.savefig(output, dpi=args.dpi)
    print(f"Saved figure to {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
