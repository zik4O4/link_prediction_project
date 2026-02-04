"""
Initialisation du package src
"""

from .data_loader import load_datasets, print_dataset_stats
from .similarity_metrics import compute_all_metrics, compute_our_measure
from .feature_extraction import prepare_dataset
from .train_models import compare_with_without_our_measure
from .visualization import plot_comparison_all_datasets, plot_accuracy_table

__all__ = [
    'load_datasets',
    'print_dataset_stats',
    'compute_all_metrics',
    'compute_our_measure',
    'prepare_dataset',
    'compare_with_without_our_measure',
    'plot_comparison_all_datasets',
    'plot_accuracy_table'
]