#!/usr/bin/env python3
"""
split_scenes.py -- host-side staging step for the CapaciNet 'scene' pipeline mode.

Run OUTSIDE the container (on the login node, in an environment with h5py --
e.g. the `capacinet` conda env from the top-level README/CLAUDE.md), NOT inside
a SLURM job: data_generation.sif has no h5py (checked directly inside the image),
and the array job itself runs with --network=none, so nothing can be installed
on the fly at job time either.

Reads one voxel grid per scene (+ voxel_size/origine_* attrs) out of a single
shared scene HDF5 and writes one lightweight .npz per scene into an output
directory. Each array task in job.sbatch then reads only its own small .npz
with plain numpy (no h5py) -- this is what makes running an array of 100-500
tasks against one big input file straightforward: they never open the shared
HDF5 concurrently, each just gets its own pre-extracted file, so there's
nothing to coordinate and no Lustre-flock() concerns to work around.

The default layout is group/<i>/voxel_grid, the same convention
data_generator.cpp's own output uses (see reachability_map.h5). If your file
uses a different layout, pass --h5-path with '{i}' standing in for the scene
index, e.g. --h5-path 'voxel_grid/{i}' for a voxel_grid/<i> layout. The group
containing '{i}' is enumerated (its numeric keys) to discover scene indices.

voxel_size and origin are looked up as attributes, tried in this order so both
the pipeline's own convention and externally-recorded files (e.g. a voxel grid
dumped straight from a ROS topic, which tends to carry these as single
file-level attributes rather than per-scene ones) work without extra flags:
  voxel_size:  dataset attr 'voxel_size' -> dataset attr 'resolution'
               -> root attr 'voxel_size' -> root attr 'resolution'
  origin:      dataset attrs 'origine_x/y/z' -> dataset attr 'origin' (3-vector)
               -> root attrs 'origine_x/y/z' -> root attr 'origin' (3-vector)
               -> symmetric default (-N*voxel_size/2 per axis)

Occupancy convention is NOT auto-detected -- pass --occupied-value explicitly
if your file doesn't use the pipeline's own convention (default -1, matching
reachability_map.h5; a file recorded straight from a ROS topic may instead use
0=occupied). This script converts to a canonical boolean 'occupied' array once,
here, so load_scene.py (running inside the container) never has to know a sign
convention at all.

Usage:
    python3 split_scenes.py --scene-file scenes.h5 --out-dir datasets/<out>/scenes
    python3 split_scenes.py --scene-file scenes.h5 --out-dir datasets/<out>/scenes \
        --h5-path 'voxel_grid/{i}' --occupied-value 0

Prints the number of scenes written, so gen_dataset.sh can size --array from it.
"""
import argparse
import os

import h5py
import numpy as np


def _first_attr(sources, names):
    """First matching attribute value, searching each attrs dict in `sources` in
    order, then each name in `names` within that dict."""
    for attrs in sources:
        for name in names:
            if name in attrs:
                return attrs[name]
    return None


def resolve_voxel_size(ds_attrs, root_attrs):
    v = _first_attr([ds_attrs, root_attrs], ["voxel_size", "resolution"])
    if v is None:
        raise SystemExit("no voxel_size/resolution attribute found on the dataset "
                          "or at the file root.")
    return float(v)


def resolve_origin(ds_attrs, root_attrs, shape, voxel_size):
    for attrs in (ds_attrs, root_attrs):
        if {"origine_x", "origine_y", "origine_z"} <= attrs.keys():
            return np.array([attrs["origine_x"], attrs["origine_y"], attrs["origine_z"]],
                             dtype=np.float64)
        if "origin" in attrs:
            return np.asarray(attrs["origin"], dtype=np.float64)
    # Symmetric default: matches the pipeline's own output when no origin is given.
    return np.array([-shape[0] * voxel_size / 2,
                      -shape[1] * voxel_size / 2,
                      -shape[2] * voxel_size / 2], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--h5-path", default="group/{i}/voxel_grid",
        help="Dataset path template for each scene's voxel grid, with '{i}' "
             "standing in for the scene index (default: group/{i}/voxel_grid, "
             "same as the pipeline's own output). Example alternative: "
             "'voxel_grid/{i}'.")
    ap.add_argument(
        "--occupied-value", type=int, default=-1,
        help="Raw value meaning 'occupied' (the other value means free). "
             "Default -1, matching the pipeline's own reachability_map.h5 "
             "output. Pass the value your file actually uses, e.g. 0.")
    args = ap.parse_args()

    if "{i}" not in args.h5_path:
        raise SystemExit(f"--h5-path must contain '{{i}}': got {args.h5_path!r}")
    prefix, _, suffix = args.h5_path.partition("{i}")
    enum_group = prefix.rstrip("/")

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.scene_file, "r") as f:
        if enum_group and enum_group not in f:
            raise SystemExit(f"'{args.scene_file}' has no group '{enum_group}' -- "
                              f"expected scenes at '{args.h5_path}'. Pass --h5-path "
                              f"if your layout differs.")
        group_obj = f[enum_group] if enum_group else f
        indices = sorted(int(k) for k in group_obj.keys() if k.isdigit())
        if not indices:
            raise SystemExit(f"no numeric-keyed scenes found under "
                              f"'{enum_group or '/'}' in '{args.scene_file}'.")

        root_attrs = dict(f.attrs)

        for i in indices:
            ds = f[f"{prefix}{i}{suffix}"]
            raw = ds[()]
            ds_attrs = dict(ds.attrs)
            voxel_size = resolve_voxel_size(ds_attrs, root_attrs)
            origin = resolve_origin(ds_attrs, root_attrs, raw.shape, voxel_size)
            occupied = (raw == args.occupied_value)
            # Compressed: occupancy grids are extremely uniform, so this is ~30x
            # smaller (8.4 MB -> ~0.3 MB for 128^3), which matters at 500 scenes.
            # Decompression cost at load time is negligible next to RM generation.
            np.savez_compressed(
                os.path.join(args.out_dir, f"scene_{i}.npz"),
                occupied=occupied.astype(np.uint8),
                voxel_size=np.float64(voxel_size),
                origin=origin,
            )

    print(len(indices))


if __name__ == "__main__":
    main()
