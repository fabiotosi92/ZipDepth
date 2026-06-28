"""Zero-shot depth evaluation for ZipDepth (Marigold protocol)."""

from zipdepth.evaluation.datasets import DATASET_CONFIGS, discover_samples, load_gt
from zipdepth.evaluation.evaluator import evaluate, print_results, save_results
from zipdepth.evaluation.metrics import (
    MetricTracker,
    compute_depth_metrics,
    METRIC_FUNCTIONS,
)

__all__ = [
    'DATASET_CONFIGS',
    'discover_samples',
    'load_gt',
    'evaluate',
    'print_results',
    'save_results',
    'MetricTracker',
    'compute_depth_metrics',
    'METRIC_FUNCTIONS',
]
