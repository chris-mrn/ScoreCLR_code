"""Evaluation, the same protocol for every method (python -m scoreclr.evaluation --run <dir>)."""
from scoreclr.evaluation.knn import extract_features, monitor, weighted_knn
from scoreclr.evaluation.linear import linear_probe

__all__ = ["extract_features", "linear_probe", "monitor", "weighted_knn"]
