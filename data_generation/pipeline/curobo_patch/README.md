# curobo_ros patches

Files mounted by `job.sbatch` over their installed copies in
`docker_images/data_generation.sif`, under
`/home/ros2_ws/install/curobo_ros/lib/python3.10/site-packages/curobo_ros/core/`.
Why this exists: `docs/adr/0001-patch-container-files-by-bind-mount.md`.

| File | Mounted in | Change against the image | Upstream status |
|---|---|---|---|
| `obstacle_manager.py` | every mode | `get_voxel_grid`: scene half extent read from `HERON_SCENE_HALF_EXTENT` (station `scene_half_extent`) instead of the hard-coded 1.52 m, which stays the default. Adds `load_voxel_scene` / `get_scene_voxel_grid` (native ESDF scenes). | not merged |
| `ros_service_manager.py` | `scene` mode | Adds the `/curobo_ik/load_voxel_scene` service and `_callback_load_voxel_scene`. | not merged |

## When the image is rebuilt

1. Copy each file above from the new image:
   `apptainer exec <new.sif> cat <installed path>/<file> > <file>`.
2. Re-apply the changes listed in the table (search for `HERON PATCH` and the added
   functions in the current copies).
3. If the change is already in the new `curobo_ros`, delete the file here, remove it
   from `job.sbatch`, and remove its row.
