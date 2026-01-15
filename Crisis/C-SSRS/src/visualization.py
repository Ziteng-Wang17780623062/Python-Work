"""
Visualization module for classification metrics
Includes confusion matrix plotting functionality
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    confusion_matrix: Dict[str, Dict[str, int]],
    model_name: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    normalize: bool = False,
    cmap: str = "Blues",
    dpi: int = 300,
) -> None:
    """
    Plot a confusion matrix heatmap.
    
    Args:
        confusion_matrix: Dictionary format {true_label: {pred_label: count}}
        model_name: Name of the model for the title
        output_path: Path to save the figure. If None, display the figure
        figsize: Figure size (width, height)
        normalize: If True, normalize the confusion matrix to percentages
        cmap: Colormap for the heatmap
        dpi: Resolution for saved figure
    """
    # Extract all labels and sort them
    # For C-SSRS, maintain the order: Supportive, Indicator, Ideation, Behavior, Attempt
    label_order = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]
    all_labels_set = set(confusion_matrix.keys()) | set(
        pred_label for row in confusion_matrix.values() for pred_label in row.keys()
    )
    
    # Sort labels: first use predefined order, then add any other labels
    all_labels = [label for label in label_order if label in all_labels_set]
    all_labels.extend(sorted(all_labels_set - set(all_labels)))
    
    if not all_labels:
        _logger.warning("No labels found in confusion matrix")
        return
    
    # Convert to numpy array
    n_classes = len(all_labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_label in enumerate(all_labels):
        for j, pred_label in enumerate(all_labels):
            count = confusion_matrix.get(true_label, {}).get(pred_label, 0)
            matrix[i, j] = count
    
    # Normalize if requested
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix.astype(float) / row_sums * 100
        fmt = '.1f'
    else:
        # Ensure matrix is integer type for integer formatting
        matrix = matrix.astype(int)
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=all_labels,
        yticklabels=all_labels,
        ax=ax,
    )
    
    # Set labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    title = 'Confusion Matrix'
    if model_name:
        title += f' - {model_name}'
    if normalize:
        title += ' (Normalized)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        _logger.info(f"Confusion matrix saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix_from_json(
    json_path: Path,
    output_dir: Optional[Path] = None,
    normalize: bool = False,
    model_name: Optional[str] = None,
) -> None:
    """
    Load classification results from JSON and plot confusion matrix.
    
    Args:
        json_path: Path to JSON file containing classification results
        output_dir: Output directory for the figure. If None, use JSON file's directory
        normalize: If True, normalize the confusion matrix to percentages
        model_name: Model name override. If None, extract from JSON
    """
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract model name
    if model_name is None:
        model_name = results.get("model", "Unknown Model")
    
    # Check for metrics (C-SSRS uses "metrics" instead of "classification_metrics")
    if "metrics" not in results:
        _logger.error(f"JSON file does not contain 'metrics' field")
        return
    
    metrics = results["metrics"]
    confusion_matrix = metrics.get("confusion_matrix")
    
    if not confusion_matrix:
        _logger.error(f"Metrics do not contain 'confusion_matrix' field")
        return
    
    # Determine output directory
    if output_dir is None:
        output_dir = json_path.parent
    
    # Generate output filename
    suffix = "_normalized" if normalize else ""
    output_file = output_dir / f"{json_path.stem}_confusion_matrix{suffix}.png"
    
    # Plot confusion matrix
    plot_confusion_matrix(
        confusion_matrix,
        model_name=model_name,
        output_path=output_file,
        normalize=normalize,
    )
    
    _logger.info(f"Confusion matrix visualization completed. Saved to: {output_dir}")
