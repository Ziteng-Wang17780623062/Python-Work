"""
C-SSRS 数据集分类任务评估入口。
使用5-label分类：Supportive, Indicator, Ideation, Behavior, Attempt
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from src.classification import run_classification, score_classification
from src.data_utils import load_dataset, save_results
from src.settings import CLASSIFICATION_MODEL, PROJECT_ROOT
from src.visualization import plot_confusion_matrix_from_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-SSRS Classification Task Evaluator")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/500_Reddit_users_posts_labels.csv"),
        help="Dataset path (CSV format), default: 500_Reddit_users_posts_labels.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results"),
        help="Output directory for results",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve relative path to absolute path relative to project root."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_sample_limit() -> int:
    """Interactive input for sample limit."""
    while True:
        try:
            user_input = input("Enter the number of samples to sample (must be a positive integer): ").strip()
            if not user_input:
                print("Sample limit cannot be empty. Please enter a positive integer.")
                continue
            limit = int(user_input)
            if limit <= 0:
                print("Sample limit must be a positive integer. Please try again.")
                continue
            return limit
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            raise


def generate_random_seed() -> int:
    """Generate a random seed based on current timestamp to ensure different samples each time."""
    # Use microsecond timestamp to ensure uniqueness
    timestamp = int(time.time() * 1000000)
    process_id = time.time_ns() % (2**32)
    random_seed = (timestamp + process_id) % (2**32)
    return random_seed


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)

    print("=" * 80)
    print("C-SSRS Classification Task Evaluator")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Model: {CLASSIFICATION_MODEL}")
    print("=" * 80)
    print()

    # Interactive input for sample limit
    sample_limit = get_sample_limit()
    
    # Generate random seed to ensure different samples each time
    random_seed = generate_random_seed()
    print(f"Using random seed: {random_seed} (ensures different samples each run)")
    print()

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    samples = load_dataset(dataset_path, limit=sample_limit, random_seed=random_seed)
    dataset_id = dataset_path.stem
    print(f"Loaded {len(samples)} samples")
    print()

    # Run classification
    print("=" * 80)
    print("Running classification task...")
    print("=" * 80)
    classification_results = run_classification(samples, dataset_id)

    # Calculate all metrics
    print()
    print("=" * 80)
    print("Calculating all evaluation metrics...")
    print("=" * 80)
    classification_metrics = score_classification(classification_results)

    # Build results
    results: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "dataset_id": dataset_id,
        "task": "classification",
        "samples": len(samples),
        "sample_limit": sample_limit,
        "random_seed": random_seed,
        "model": CLASSIFICATION_MODEL,
        "classification": classification_results,
        "metrics": classification_metrics,
    }

    # Save results to a single file
    print()
    print("=" * 80)
    print("Saving results...")
    print("=" * 80)
    output_path = save_results(results, output_dir)
    print(f"\nAll results saved to: {output_path}")
    
    # Generate confusion matrix visualization if metrics are available
    if "metrics" in results:
        try:
            print()
            print("=" * 80)
            print("Generating confusion matrix visualization...")
            print("=" * 80)
            plot_confusion_matrix_from_json(
                output_path,
                output_path.parent,
                normalize=False,
            )
            # Also generate normalized version
            plot_confusion_matrix_from_json(
                output_path,
                output_path.parent,
                normalize=True,
            )
            print("Confusion matrix visualization completed!")
        except ImportError as e:
            print(f"Warning: Cannot generate visualization. Please ensure matplotlib and seaborn are installed: {e}")
        except Exception as e:
            print(f"Warning: Error occurred while generating confusion matrix: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
