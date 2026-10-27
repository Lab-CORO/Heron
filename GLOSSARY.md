# Heron

Heron predicts, for a robot in a cluttered workspace, which points it can reach. It generates reachability datasets for a robot and a station, then trains and evaluates a 3D U-Net on them.

## Language

### Setup

**Robot**:
An arm as Heron sees it: its kinematic description, its base frame and how far it reaches. Independent of where it is installed.
_Avoid_: arm config, robot env

**Station**:
A physical installation where a Robot works: its fixed obstacles, its cameras and the region around the base that a Scene covers. One Robot can work in several Stations.
_Avoid_: setup, cell, world (when meaning the whole installation)

**Site**:
A machine or cluster Heron runs on, with what differs between them: how jobs are launched, the accounts, the modules, the environments and images, where the data lives.
_Avoid_: platform, system, cluster (a cluster is one kind of Site)

### Data

**Master map**:
The reachability of a Robot with no obstacles, computed once per Robot and voxel size. Every Reachability map is derived from it.
_Avoid_: master IK data, RM (ambiguous)

**Scene**:
The occupancy of a Station at one moment, as a voxel grid: which voxels hold an obstacle.
_Avoid_: voxel grid (when meaning the content), environment

**Reachability map**:
For one Scene, how reachable each voxel is for the Robot, as a value between 0 and 1.
_Avoid_: RM, label (outside training code)

**Voxel size**:
The edge length of one voxel in meters, chosen per Dataset. Master map, Scenes and Reachability maps of a Dataset share it.
_Avoid_: resolution

**Dataset**:
A set of Scene / Reachability map pairs generated for one Robot in one Station at one Voxel size.

**Source**:
What a Dataset's Scenes were captured from: one rosbag, one generated file of random obstacles, or one supplied Scene. Scenes from the same Source are near duplicates.

**Corpus**:
One or more Datasets made ready for learning: cropped, cut by Source into train, validation and named test parts, with only the train part augmented. An Experiment trains and is evaluated on a Corpus.
_Avoid_: training set (that is the train part), prepared dataset, formatted data

**Crop**:
Cutting every Scene and Reachability map of a Dataset to a smaller cube around the base, so it matches the network's input size.

### Learning

**Experiment**:
One training run of the network on a Corpus, with its recipe, its checkpoints and its evaluation results.
_Avoid_: run, job (a job is only the cluster submission)

**Package**:
An Experiment made portable for inference on another machine: its config, one checkpoint, the network as ONNX, and a sample of its train volumes for int8 calibration.
_Avoid_: export (that is the command), bundle, model (the network alone)

**Engine**:
The network compiled by TensorRT at one precision (fp32, fp16, int8), for one GPU and one TensorRT version. Built from a Package on the machine that runs it.
_Avoid_: TRT model, plan
