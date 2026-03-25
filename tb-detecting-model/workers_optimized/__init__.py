"""
Workers Optimization Package

This package contains all worker-optimized training and feature extraction code.

Usage:
  from workers_optimized.pipeline_workers import extract_features_with_workers
  from workers_optimized.train_with_workers import main
"""

__version__ = "1.0.0"
__author__ = "TB Detection System"

from .pipeline_workers import (
    WorkerConfig,
    extract_features_with_workers,
    extract_features_batched,
    extract_features_for_single_image,
    configure_sklearn_workers,
    get_dataloader_config,
    print_worker_diagnostics
)

__all__ = [
    'WorkerConfig',
    'extract_features_with_workers',
    'extract_features_batched',
    'extract_features_for_single_image',
    'configure_sklearn_workers',
    'get_dataloader_config',
    'print_worker_diagnostics'
]
