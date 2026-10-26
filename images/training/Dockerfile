# Training image for heron's apptainer runtime (amd64). Build on a machine with Docker,
# then convert to Apptainer and copy the .sif where site.yaml points
# (images.amd64.training):
#
#   docker build -f images/training/Dockerfile -t heron-training:1.9.6 .
#   apptainer build heron_training.sif docker-daemon://heron-training:1.9.6
#
# Requirements are in requirements-train.txt; docs/images.md lists what the image
# must provide, and `heron doctor` checks it.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

COPY requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt && rm /tmp/requirements-train.txt

# heron itself is not installed: jobs mount the repository and set PYTHONPATH.
