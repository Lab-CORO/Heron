# Patch curobo_ros by bind-mounting files over the image

Status: accepted

The data-generation image (`docker_images/data_generation.sif`, 18 GB) takes long to
rebuild and is shared by every run on the cluster. When a pure-Python file of
`curobo_ros` inside it has to change (the native ESDF scene service, the station's scene
extent), we keep a modified copy in `data_generation/pipeline/curobo_patch/` and
`job.sbatch` bind-mounts it over the installed file, instead of rebuilding the image.
Each copy starts from the file **as installed in the image** (not from the
`curobo_ros` workspace, which has diverged) and only adds or changes what Heron needs.

## Consequences

- A rebuilt image with a newer `curobo_ros` silently loses its upstream changes to any
  patched file, because the mount wins. Before switching images, re-copy each file from
  the new image and re-apply the changes listed in `curobo_patch/README.md`.
- Compiled code (the C++ generator) cannot be patched this way: the master map radius
  stays hard-coded at 2 m until the next image rebuild.
- We leave this approach once the changes are merged into `curobo_ros` and the image is
  rebuilt; the inventory in `curobo_patch/README.md` is the list to empty.
