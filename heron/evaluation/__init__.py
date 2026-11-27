"""Evaluation of an Experiment on its Corpus: val and every named test part.

measure.py runs inside the training image (same pytorch-3dunet as training) and
writes eval_data.h5, report.txt and the results summary; figures.py draws the
figures on the host from eval_data.h5.
"""
