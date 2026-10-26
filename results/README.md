# Results

One `<experiment>.yaml` per evaluated Experiment, written by `heron evaluate` (or the
evaluation chained by `heron train`): recipe, corpus and its input fingerprint, git commit,
checkpoint, and for each part (val, test:<name>) the regression, 51-class and per-region
metrics. Commit them; the figures and caches stay in the experiment directory.
