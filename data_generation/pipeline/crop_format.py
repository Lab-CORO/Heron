"""
crop_format.py — standalone cropper + unet3d formatter.

Converts generated CapaciNet HDF5 (group/N/{voxel_grid, reachability_map}) — or
already-formatted raw/label files — into unet3d-compatible files, center-cropped to
a target cube size (e.g. 152^3 -> 128^3).

Modeled on unet_3d/script/format_data.py, but focused on cropping: it computes the
centered crop automatically from --size, updates the geometry attributes
(origine_*, voxel_grid_size_*), and is independent of the unet_3d package so it can
be run on its own.

Cropping is controlled by explicit indices (--crop x0 x1 y0 y1 z0 z1, default
12 140 12 140 20 148 — same as launch_training.py) or, alternatively, by a centered
cube size (--size 128, which overrides --crop).

Example:
    python crop_format.py --input_dir datasets/real_long \
        --output_dir CapaciNet/unet_3d/data/real_long_128 --split
"""
import argparse
import os
import pathlib
import random
import shutil

import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # keep the script usable without tqdm installed
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []
    tqdm.write = print


def _open_read(filepath):
    """Open an HDF5 file read-only, disabling locking (needed on Lustre)."""
    try:
        return h5py.File(filepath, "r", locking=False)
    except TypeError:  # older h5py without the locking kwarg
        return h5py.File(filepath, "r")


class DatasetCropper:
    """Center-crop voxel volumes to a target size and save in unet3d (raw/label) format."""

    def __init__(self, input_dir, output_dir, size=None, crop=None, drop_empty=True):
        """
        Args:
            input_dir (str): Directory containing input HDF5 files.
            output_dir (str): Directory where cropped/formatted files are written.
            size (int | None): If set, do a centered crop to a size^3 cube (overrides `crop`).
            crop (tuple | None): Explicit ((x0, x1), (y0, y1), (z0, z1)) crop indices.
            drop_empty (bool): Skip samples whose reachability_map (label) is all zeros.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.size = int(size) if size is not None else None
        self.crop = crop
        self.drop_empty = drop_empty

    def _resolve_slices(self, shape):
        """Return per-axis (start, stop) slices for this file's shape.

        Priority: centered `size` if set, else explicit `crop` indices, else no crop.

        Raises:
            ValueError: if the requested crop does not fit the volume.
        """
        if self.size is not None:
            slices = []
            for axis, dim in enumerate(shape):
                if self.size > dim:
                    raise ValueError(f"size {self.size} exceeds volume dim {dim} on axis {axis}")
                start = (dim - self.size) // 2
                slices.append((start, start + self.size))
            return slices
        if self.crop is not None:
            for axis, ((a0, a1), dim) in enumerate(zip(self.crop, shape)):
                if not (0 <= a0 < a1 <= dim):
                    raise ValueError(f"crop [{a0}:{a1}] out of range for dim {dim} on axis {axis}")
            return self.crop
        return [(0, dim) for dim in shape]

    def _crop(self, voxel_grid, reachability_map, attrs):
        """Crop both volumes (per _resolve_slices) and update geometry attributes."""
        (x0, x1), (y0, y1), (z0, z1) = self._resolve_slices(voxel_grid.shape)
        voxel_grid = voxel_grid[x0:x1, y0:y1, z0:z1]
        reachability_map = reachability_map[x0:x1, y0:y1, z0:z1]

        attrs = dict(attrs)
        voxel_size = float(attrs.get("voxel_size", 0.0))
        # Shift the origin by the crop offset so the world geometry stays consistent.
        for key, start in (("origine_x", x0), ("origine_y", y0), ("origine_z", z0)):
            if key in attrs and voxel_size:
                attrs[key] = float(attrs[key]) + start * voxel_size
        for key, (a0, a1) in (("voxel_grid_size_x", (x0, x1)),
                              ("voxel_grid_size_y", (y0, y1)),
                              ("voxel_grid_size_z", (z0, z1))):
            if key in attrs:
                attrs[key] = float(a1 - a0)
        return voxel_grid, reachability_map, attrs

    def _save(self, voxel_grid, reachability_map, filename, attrs):
        """Write voxel_grid as 'raw' and reachability_map as 'label' (gzip)."""
        with h5py.File(filename, "w") as f:
            raw = f.create_dataset("raw", data=voxel_grid, compression="gzip")
            label = f.create_dataset("label", data=reachability_map, compression="gzip")
            for key, val in attrs.items():
                raw.attrs[key] = val
                label.attrs[key] = val

    def _emit(self, voxel_grid, reachability_map, attrs, out_name):
        """Crop, optionally drop empty, and save one sample. Returns True if written."""
        if self.drop_empty and not np.any(reachability_map):
            return False
        voxel_grid, reachability_map, attrs = self._crop(voxel_grid, reachability_map, attrs)
        self._save(voxel_grid, reachability_map, os.path.join(self.output_dir, out_name), attrs)
        return True

    def run(self):
        """Process every .h5 in input_dir and write cropped/formatted files to output_dir."""
        os.makedirs(self.output_dir, exist_ok=True)
        h5_files = [f for f in os.listdir(self.input_dir) if f.endswith(".h5")]

        written, dropped, skipped = 0, 0, []
        for file_name in tqdm(h5_files, desc="Files", unit="file"):
            filepath = os.path.join(self.input_dir, file_name)
            stem = os.path.splitext(file_name)[0]
            try:
                with _open_read(filepath) as f:
                    is_formatted = "raw" in f and "label" in f
                    if is_formatted:
                        samples = [(file_name, f["raw"][:], f["label"][:], dict(f["raw"].attrs))]
                    elif "group" in f:
                        samples = []
                        for index in sorted(f["group"].keys(), key=int):
                            grp = f["group"][index]
                            if "voxel_grid" not in grp or "reachability_map" not in grp:
                                continue
                            samples.append((
                                f"{stem}_group_{index}.h5",
                                grp["voxel_grid"][:],
                                grp["reachability_map"][:],
                                dict(grp["reachability_map"].attrs),
                            ))
                    else:
                        tqdm.write(f"WARNING: no 'group' or 'raw/label' in {file_name}, skipping")
                        skipped.append(file_name)
                        continue
            except OSError as e:
                tqdm.write(f"WARNING: skipping corrupted file {file_name}: {e}")
                skipped.append(file_name)
                continue

            for out_name, voxel_grid, reachability_map, attrs in samples:
                if self._emit(voxel_grid, reachability_map, attrs, out_name):
                    written += 1
                else:
                    dropped += 1

        print(f"\nDone: {written} written, {dropped} dropped (empty), {len(skipped)} files skipped.")
        if skipped:
            print(f"Skipped files: {skipped}")

    def split(self, val_ratio=0.1, seed=0):
        """Move output files into train/ and val/ subdirectories."""
        root = pathlib.Path(self.output_dir)
        files = sorted(root.glob("*.h5"))
        (root / "train").mkdir(parents=True, exist_ok=True)
        (root / "val").mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        val_files = set(random.sample(files, max(1, int(len(files) * val_ratio)))) if files else set()
        for f in files:
            dest = root / ("val" if f in val_files else "train") / f.name
            shutil.move(str(f), str(dest))
        print(f"Split done: {len(files) - len(val_files)} train, {len(val_files)} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center-crop CapaciNet HDF5 and format for unet3d.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with generated/raw HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for cropped unet3d files")
    parser.add_argument(
        "--crop", type=int, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        default=[12, 140, 12, 140, 20, 148],
        help="Explicit crop indices x0 x1 y0 y1 z0 z1 (default: 12 140 12 140 20 148)",
    )
    parser.add_argument("--size", type=int, default=None,
                        help="Centered crop to size^3 instead of explicit --crop indices")
    parser.add_argument("--keep-empty", dest="drop_empty", action="store_false",
                        help="Keep samples whose reachability_map is all zeros (default: drop them)")
    parser.add_argument("--split", action="store_true", help="Split output into train/ and val/ subfolders")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation fraction for --split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --split (default: 0)")
    args = parser.parse_args()

    # --size (centered) takes precedence; otherwise use the explicit --crop indices.
    crop = None
    if args.size is None:
        c = args.crop
        crop = ((c[0], c[1]), (c[2], c[3]), (c[4], c[5]))

    cropper = DatasetCropper(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size,
        crop=crop,
        drop_empty=args.drop_empty,
    )
    cropper.run()
    if args.split:
        cropper.split(val_ratio=args.val_ratio, seed=args.seed)
