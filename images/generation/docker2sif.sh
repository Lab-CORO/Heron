#!/usr/bin/env bash
#
# docker2sif.sh — convert a Docker image to a .sif with about twice the squashfs size of
# free disk, where `apptainer build docker-daemon://` needs about three times the image size
# (a docker-save tar, its unpacked rootfs, then the squashfs).
#
#   images/generation/docker2sif.sh <image:tag> <out.sif> <work dir>
#
# The container's filesystem is streamed (docker export) straight into a squashfs (sqfstar),
# together with the /.singularity.d tree an Apptainer build would add: its standard scripts
# (taken from an empty `Bootstrap: scratch` build), the image's ENV as
# env/10-docker2singularity.sh and its labels in labels.json. The squashfs then becomes the
# .sif's system partition. The result runs and inspects like an `apptainer build` one.
set -euo pipefail

IMAGE="$1"; SIF="$2"; WORK="$3"
die() { echo "docker2sif.sh: $*" >&2; exit 1; }
for t in docker apptainer sqfstar python3; do command -v "$t" >/dev/null || die "$t not found"; done
docker image inspect "$IMAGE" >/dev/null || die "no Docker image $IMAGE"

rm -rf "$WORK"; mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
trap 'docker rm -f "${cid:-}" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

# Standard /.singularity.d, from an empty build.
printf 'Bootstrap: scratch\n' > "$WORK/skel.def"
APPTAINER_TMPDIR="$WORK" apptainer --quiet build --fakeroot --sandbox "$WORK/skel" "$WORK/skel.def"
mkdir -p "$WORK/extra"
(cd "$WORK/skel" && cp -a .singularity.d .exec .run .shell .test singularity environment "$WORK/extra/")

# The image's ENV and labels, in the form apptainer's own Docker conversion writes them.
docker image inspect "$IMAGE" --format '{{json .Config}}' > "$WORK/config.json"
python3 - "$WORK" "$IMAGE" <<'PY'
import json, os, sys
work, image = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{work}/config.json"))
lines = ["#!/bin/sh"]
for kv in cfg.get("Env") or []:
    k, v = kv.split("=", 1)
    v = v.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    # PATH is set outright; the rest keeps a value given with APPTAINERENV_<name>.
    lines.append(f'export {k}="{v}"' if k == "PATH" else f'export {k}="${{{k}:-"{v}"}}"')
env = f"{work}/extra/.singularity.d/env/10-docker2singularity.sh"
with open(env, "w") as f:
    f.write("\n".join(lines) + "\n")
os.chmod(env, 0o755)
path = f"{work}/extra/.singularity.d/labels.json"
labels = json.load(open(path))
labels.update(cfg.get("Labels") or {})
labels["org.label-schema.usage.singularity.deffile.bootstrap"] = "docker-daemon"
labels["org.label-schema.usage.singularity.deffile.from"] = image
with open(path, "w") as f:
    json.dump(labels, f, indent=1, sort_keys=True)
PY

# Filesystem + /.singularity.d -> squashfs, in one stream.
cid="$(docker create "$IMAGE")"
{ docker export "$cid"; tar -C "$WORK/extra" --owner=0 --group=0 -cf - .; } \
  | sqfstar -ignore-zeros -quiet "$WORK/rootfs.sqfs"
docker rm "$cid" >/dev/null; cid=""

# squashfs -> .sif (system partition, amd64), as apptainer build lays it out.
rm -f "$SIF"
apptainer sif new "$SIF"
apptainer sif add --datatype 4 --parttype 2 --partfs 1 --partarch 2 --groupid 1 "$SIF" "$WORK/rootfs.sqfs"
apptainer sif list "$SIF"
