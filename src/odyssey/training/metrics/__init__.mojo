"""
Training Metrics.

Metric implementations for tracking training and evaluation performance

Includes:
- Accuracy (Classification accuracy - top-1, top-k, per-class)
- LossTracker (Loss tracking and averaging)
- ConfusionMatrix (Confusion matrix for classification)
- CSVMetricsLogger (CSV-based training metrics logging)
- Precision (Precision metric)
- Recall (Recall metric)

All metrics implement the Metric trait for consistent interface

Note:
    All symbols in this module are re-exported cleanly through the parent
    `odyssey.training` package. You may import directly from either location:

    ```mojo
    from odyssey.training.metrics import AccuracyMetric
    from odyssey.training import AccuracyMetric  # also works
    ```

    No Mojo re-export limitation applies here (unlike `odyssey.training.callbacks`).
"""

# Export base metric interface and utilities
from odyssey.training.metrics.base import (
    Metric,
    MetricResult,
    MetricCollection,
    MetricLogger,
    create_metric_summary,
)

# Export metric implementations
from odyssey.training.metrics.accuracy import (
    top1_accuracy,
    topk_accuracy,
    per_class_accuracy,
    AccuracyMetric,
)
from odyssey.training.metrics.loss_tracker import (
    LossTracker,
    Statistics,
    ComponentTracker,
)
from odyssey.training.metrics.confusion_matrix import ConfusionMatrix

# Consolidated evaluation utilities
from odyssey.training.metrics.evaluate import (
    evaluate_with_predict,
    evaluate_logits_batch,
    compute_accuracy_on_batch,
)

# Results printing utilities
from odyssey.training.metrics.results_printer import (
    print_evaluation_summary,
    print_per_class_accuracy,
    print_confusion_matrix,
    print_training_progress,
    print_training_summary,
)

# CSV-based training metrics logging
from odyssey.training.metrics.csv_metrics_logger import CSVMetricsLogger

# Future exports (to be implemented):
# from .precision import Precision
# from .recall import Recall
