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

**Crop**:
Cutting every Scene and Reachability map of a Dataset to a smaller cube around the base, so it matches the network's input size.

### Learning

**Experiment**:
One training run of the network on a Dataset, with its recipe, its checkpoints and its evaluation results.
_Avoid_: run, job (a job is only the cluster submission)
