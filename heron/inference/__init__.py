"""Inference outside training: a portable package of an Experiment, its TensorRT engines, their timing.

    heron export <experiment>                     package: config, checkpoint, ONNX, int8 calibration sample
    heron benchmark <package> --precision ...     on any machine: build its engines, time them
    heron evaluate <experiment> --precision ...   on the cluster: timing plus quality per precision
    heron.inference.load(package, precision)      in a program (a robot): the package's network, ready to predict

Engines are built on the machine that runs them (a TensorRT engine is tied to its GPU and
TensorRT version) and cached in the package's engines/ directory, never committed.
"""

from heron.inference.predictor import Predictor, load  # noqa: E402,F401
