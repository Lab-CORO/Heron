# Results

One `<experiment>.yaml` per evaluated Experiment, written by `heron evaluate` (or the
evaluation chained by `heron train`): recipe, corpus and its input fingerprint, git commit,
checkpoint, and for each part (val, test:<name>) the regression, 51-class and per-region
metrics. Commit them; the figures and caches stay in the experiment directory.

With `heron evaluate --precision`, the same file also holds the timing of PyTorch and of
each TensorRT engine on the cluster's GPU, and each engine's metrics with their delta from
PyTorch's. `heron benchmark` writes `<experiment>/benchmark_<gpu>.yaml`, one per machine
timed (an A100, a Jetson Orin, ...): timings only, with the GPU and software versions.
