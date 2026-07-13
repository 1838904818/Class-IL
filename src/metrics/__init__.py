"""Evaluation metrics for Class-IL."""
from src.metrics.accuracy import avg_accuracy, avg_forgetting, evaluate_per_class, evaluate_task

__all__ = ["avg_accuracy", "avg_forgetting", "evaluate_per_class", "evaluate_task"]
