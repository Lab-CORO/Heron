# Unified data-generation pipeline

One command to generate CapaciNet datasets — **real** (from a rosbag),
**synthetic**, or **scene** (from occupancy grids you already have), for **any
robot** (leeloo/m1013, ur5e, jaco7, iiwa…) — and to (re)build a robot's master
reachability map.

```
data_generation/pipeline/
  gen_dataset.sh      # the single entry point (run on a login node; submits SLURM)
  job.sbatch          # generic SLURM job (don't run directly)
  run_generation.sh   # in-container launcher (staged into the data dir)
  robots/<name>.env   # robot registry: robot_config_file, base_link, master_ik dir, world
  split_scenes.py     # (scene mode) host-side: scenes.h5 -> one .npz per scene
  load_scene.py       # (scene mode) in-container: inject scene into curobo, trigger RM
  curobo_patch/       # (scene mode) curobo_ros files bind-mounted over the .sif's copies
  verify_voxel_esdf.py / .sbatch   # one-off GPU check of the native ESDF path
```

## Usage

```bash
# Real dataset from a rosbag (camera extrinsics resolved from /tf):
CapaciNet/data_generation/pipeline/gen_dataset.sh \
    --robot leeloo --mode real \
    --rosbag /path/rosbag2_2026_06_02-20_58_42_0.db3 \
    --camera-frame <camera_optical_frame> \
    --array 100 --dataset-size 10 --out real_long

# Synthetic dataset for another robot:
gen_dataset.sh --robot ur5e --mode synthetic --array 500 --obj-max 20 --out ur5e_syn

# (Re)generate a robot's master_ik_data<voxel>.npz (run once per robot/voxel):
gen_dataset.sh --robot jaco7 --mode create_rm --voxel 0.02

# Dataset from scenes you already have as occupancy grids (one array task per scene):
gen_dataset.sh --robot leeloo --mode scene \
    --scene-file /path/scenes.h5 --out scene_batch1
```

Add `--dry-run` to print the `sbatch` command and the staging steps without
submitting or touching the filesystem. `--help` lists all options.

### Key options
`--robot` `--mode {real|synthetic|scene|create_rm}` `--out <name>` `--rosbag <path>`
`--camera-frame <frame>` `--scene-file <path>` `--scene-h5-path <template>`
`--scene-occupied-value <int>` `--array <N|a-b>` `--voxel 0.02` `--dataset-size`
`--batch-size` `--init-batch-size` `--obj-max` `--reach-max` `--time` `--sif`.

## How it works

`gen_dataset.sh` sources `robots/<robot>.env`, validates inputs, stages a per-run
**output directory** under `datasets/<out>/` (the robot `master_ik` is **hardlinked**
in — instant, no extra disk), stages the rosbag to `$SCRATCH/rosbag/` for `real`,
and submits `job.sbatch` as an array, passing every parameter via `--export`.

`job.sbatch` bind-mounts the host files **over the container's installed paths** (no
`.sif` rebuild): the output dir → `…/share/data_generation/data`, the parametrized
launches → `…/share/data_generation/launch`. Because apptainer runs with `-C`
(`--contain`), host env vars are **not** visible inside the container, so the
parameters are forwarded with the `APPTAINERENV_` prefix. It then runs
`run_generation.sh`, which
picks the right `ros2 launch` for the mode and forwards `robot_config_file`,
`base_link`, `world_file`, voxel/dataset params (and for `real`, the TF cameras config
+ `bag_file`). Output HDF5 is written by `data_generator.cpp` into the bound output
directory.

## Real mode: camera calibration via TF

The camera pose is resolved from the rosbag's **TF tree** (the camera moved between
recordings, so the old hardcoded extrinsics are not reused):

- `config/cameras.tf.yaml` is a template with `extrinsics: []`, which makes `curobo_ros`
  look up the transform from the robot base to the camera frame
  (`camera_strategy.py:_parse_extrinsics` → `camera_depth_map_strategy.py`).
- `--camera-frame` is substituted into `frame_id` at staging time, producing the
  explicit `datasets/<out>/cameras.yaml` used by the job.
- `generate_real_data.launch.py` replays `/tf` and `/tf_static` (in addition to the
  two `/masked_depth_image/*` topics) so the lookup resolves.

> **Base frame.** `camera_depth_map_strategy.py` looks up the transform against a frame
> literally named **`base_link`**. For **leeloo** this is fine: `base_link` and the IK
> base `base_0` are coincident frames, so the camera pose is correct. For a **different
> robot** where `base_link` does **not** coincide with the curobo robot base, the
> reconstructed obstacles would be offset — in that case make the camera lookup use the
> robot's base (bind-override `camera_depth_map_strategy.py`). Either way, run a
> **single test task first** (`--array 1`, small `--dataset-size`) and check the
> obstacle/scene alignment in the output HDF5 before launching the full array.

## Scene mode: RMs for scenes you already have

Use this when the environment is already known as a **dense occupancy voxel grid**
(not a rosbag to replay, not random cubes). One array task = one scene.

```bash
gen_dataset.sh --robot leeloo --mode scene --scene-file scenes.h5 --out scene_batch1
```

By default, `scenes.h5` is expected to follow the same layout the pipeline
already writes: `group/<i>/voxel_grid` with `voxel_size` / `origine_x|y|z`
attributes, and **occupancy convention `-1` = occupied, `1` = free** (matches
`reachability_map.h5`). Real files often differ — override with:

- `--scene-h5-path <template>`: dataset path per scene, `{i}` standing in for
  the scene index, e.g. `--scene-h5-path 'voxel_grid/{i}'` for a
  `voxel_grid/<i>` layout instead of `group/<i>/voxel_grid`. The group
  containing `{i}` is enumerated (its numeric keys) to discover scene indices.
  `voxel_size`/`origin` are looked up first as attributes on the matched
  dataset, then falling back to file-root attributes (`resolution` / `origin`
  as a 3-vector, or `origine_x|y|z`) — useful for files recorded straight from
  a ROS topic, which tend to store these once for the whole file rather than
  per scene.
- `--scene-occupied-value <int>`: the raw value meaning "occupied" (default
  `-1`). **Not auto-detected** — check your file (e.g. via `h5py`,
  `f.attrs`/`ds.attrs`) and pass the right value explicitly, e.g. `0` for a
  file using the `0=occupied` convention from the `data_generator.cpp:209`
  comment.

```bash
gen_dataset.sh --robot leeloo --mode scene --scene-file scenes.h5 \
    --scene-h5-path 'voxel_grid/{i}' --scene-occupied-value 0 --out scene_batch1
```

Both are handled entirely by `split_scenes.py` on the host, before the job is
submitted: it resolves the source convention once and writes a canonical
boolean `occupied` array into each scene's `.npz`, so `load_scene.py`
(in-container) never has to guess a sign convention, and never sees the
original HDF5 at all.

`--array` is set automatically from the number of scenes found; pass it explicitly
only to run a subset (e.g. `--array 1` to smoke-test one scene first).

**Runtime.** One scene = one RM = `len(master_ik_data)/batch_size` IK service calls
— about 4000 batches at voxel 0.02, ~1.2 s each, so **~1.5 h per scene** (measured,
A100) on top of ~2 min of cuRobo startup. `--time` defaults to `4:00:00`; raise it
if you lower `--batch-size`. Each array task does exactly one scene, so tasks run in
parallel and total wall time is set by the queue, not by the scene count.

### How the scene reaches cuRobo

`load_scene.py` tries two routes, in order:

1. **Native ESDF (default).** The grid becomes a `curobo.geom.types.VoxelGrid`
   (`WorldConfig.voxel` + `CollisionCheckerType.VOXEL`) — cuRobo's own ESDF
   obstacle type — via a new `/curobo_ik/load_voxel_scene` service. **No cuboid
   decomposition at all**, so scene complexity costs nothing extra. Verified on GPU
   by `verify_voxel_esdf.py` (submit `verify_voxel_esdf.sbatch`): an externally
   supplied ESDF array *is* enforced by `WorldVoxelCollision` (+0.15 inside a solid
   blob, −1.0 in free space).
2. **Cuboid fallback.** If that service is missing or fails, the grid is merged into
   axis-aligned boxes (exact — box volume equals occupied volume) and pushed through
   the existing `/add_object`. Only viable for blocky scenes: on a noisy 64³ test
   scene it still produced ~2000 boxes, well past the default `{"obb": 100}`
   collision cache. It is a safety net, not the intended route.

### No `.sif` rebuild required

`curobo_ros` is pure Python, so route 1 ships as two files in `curobo_patch/`
(`obstacle_manager.py`, `ros_service_manager.py`, copied from the image and
extended). `job.sbatch` bind-mounts them over the container's installed copies —
same trick already used for `DATA_DIR` / `LAUNCH_DIR`. Nothing is recompiled.

> **Keep the patch in sync.** These files are forks of the versions inside
> `docker_images/data_generation.sif`. If that image is ever rebuilt with a newer
> `curobo_ros`, re-copy them from the new image and re-apply `load_voxel_scene` /
> `get_scene_voxel_grid` / `_callback_load_voxel_scene`, or the bind-mount will
> silently revert unrelated upstream fixes.

### Why the HDF5 is split up front

`data_generation.sif` has **no h5py**, and array tasks run `--network=none`, so
nothing can be installed at job time. `gen_dataset.sh` therefore runs
`split_scenes.py` on the login node (needs the `capacinet` conda env) to write
`datasets/<out>/scenes/scene_<i>.npz`; each task then reads only its own file with
plain numpy. This also sidesteps concurrent reads of one shared file on Lustre
entirely — there is no shared handle to contend over, and no `HDF5_USE_FILE_LOCKING`
workaround needed.

## Post-processing (after generation)

Raw HDF5 (`group/N/{voxel_grid, reachability_map}`) lands in `datasets/<out>/`.

### Crop + format to unet3d (`crop_format.py`)

Standalone script: crops each volume (152³ → 128³) and writes unet3d `raw`/`label`
files (updating the `origine_*` / `voxel_grid_size_*` attributes, dropping all-zero
reachability maps, robust to Lustre file locks).

Cropping is by **explicit indices** `--crop x0 x1 y0 y1 z0 z1` (default
`12 140 12 140 20 148`, same as `launch_training.py` — a 128³ window shifted in z to
drop the floor), or by a **centered** cube via `--size N` (overrides `--crop`).

```bash
python CapaciNet/data_generation/pipeline/crop_format.py \
    --input_dir datasets/real_long \
    --output_dir CapaciNet/unet_3d/data/real_long_128 \
    [--crop 12 140 12 140 20 148] [--size 128] [--split --val_ratio 0.1] [--keep-empty]
```

For upsampling / Z-rotation augmentation, use `unet_3d/script/format_data.py` instead.

### Two-bucket split (validation réelle + entraînement augmenté)

```bash
# 1) drop empty / failed reachability maps
python CapaciNet/unet_3d/script/clean_dataset.py ...   # -> rejected/

# 2a) hold out a fraction as a pure REAL validation set (no augmentation)
python CapaciNet/unet_3d/script/format_data.py --input_dir <subset> --output_dir <val_real>

# 2b) augment the rest for training (Z-rotations, no split)
python CapaciNet/unet_3d/script/format_data.py --input_dir <rest> --output_dir <train_aug> --n_copies K
```

Then point `unet_3d/config/train_reach.yaml` (`train`/`val` dirs) at those folders.

> **Array filename note:** `data_generator.cpp` names output by timestamp (second
> resolution). With large arrays two tasks can collide (cf. the old `data_corrompu/`).
> The random start delay in `run_generation.sh` diversifies the captured scene (its
> real purpose), it is **not** collision avoidance — rely on `clean_dataset.py` and an
> HDF5 integrity check to drop any corrupt file.

## Migration

This pipeline supersedes the per-robot scripts at the project root:
`gpu_job*.sh`, `data_*/gpu_job_*.sh`, `create_rm_ur5e.sh`,
`real_data_generation_job.sh`, `data/main.bash`, `data/run_real_data_generation.sh`,
and the `create_reachability_map_<robot>.launch.py` variants. Remove them once the new
pipeline is validated.
