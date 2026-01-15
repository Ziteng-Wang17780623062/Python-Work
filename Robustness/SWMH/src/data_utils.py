"""Utilities for loading SWMH data and saving results."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _normalize_label_from_dataset(label: str) -> str:
    """
    Normalize dataset labels (e.g., self.Anxiety) into standard labels.
    """
    if not label or pd.isna(label):
        return "offmychest"
    
    label_str = str(label).strip()
    
    # Remove "self." prefix
    if label_str.startswith("self."):
        label_str = label_str[5:]
    
    # Lowercase and standardize
    label_str = label_str.lower()
    
    # Keep only these labels
    label_mapping = {
        "depression": "depression",  # depression (34.29%)
        "suicidewatch": "suicidal",  # SuicideWatch -> suicidal (18.81%)
        "anxiety": "anxiety",  # Anxiety (17.62%)
        "offmychest": "offmychest",  # offmychest (15.12%)
        "bipolar": "bipolar",  # bipolar (14.16%)
    }
    
    # If not in mapping, keep lowercase label
    return label_mapping.get(label_str, label_str)


def load_dataset(
    dataset_path: Path, 
    limit: Optional[int] = None, 
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load SWMH dataset (CSV). Returns normalized samples.
    """
    import os
    import time
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if random_seed is None:
        # Combine high-precision timestamp and PID for variability
        timestamp = int(time.time() * 1000000)  # microsecond timestamp
        process_id = os.getpid()
        random_seed = (timestamp + process_id) % (2**32)
        print(f"Auto-generated random seed: {random_seed}")
    else:
        print(f"Using provided random seed: {random_seed}")
    
    df = pd.read_csv(dataset_path, encoding="utf-8")
    
    print(f"Raw dataset size: {len(df)} rows")
    
    text_column = None
    label_column = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["text", "post", "content", "message", "input"]:
            text_column = col
        elif col_lower in ["label", "labels", "class", "category"]:
            label_column = col
    
    if text_column is None:
        text_column = df.columns[0]
    
    if label_column is None:
        label_column = df.columns[-1]
    
    print(f"Using text column: {text_column}, label column: {label_column}")
    
    df = df[df[text_column].notna()]
    df = df[df[text_column].astype(str).str.strip() != ""]
    
    df = df[df[label_column].notna()]
    df = df[df[label_column].astype(str).str.strip() != ""]
    
    print(f"Dataset size after cleaning: {len(df)} rows")
    
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=random_seed).reset_index(drop=True)
        print(f"Dataset size after sampling: {len(df)} rows (seed: {random_seed})")
    
    samples = []
    for idx, row in df.iterrows():
        text = str(row[text_column]).strip()
        raw_label = str(row[label_column]).strip()
        
        label = _normalize_label_from_dataset(raw_label)
        
        inputs = [text]
        
        sample = {
            "inputs": inputs,
            "label": label,
            "text": text,
            "raw_label": raw_label,
            "index": idx,
        }
        
        for col in df.columns:
            if col not in [text_column, label_column]:
                sample[f"meta_{col}"] = row[col]
        
        samples.append(sample)
    
    label_counts = {}
    for sample in samples:
        label = sample["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    return samples


def save_results(results: Dict[str, Any], output_dir: Path, filename_base: Optional[str] = None) -> Path:
    """
    Save evaluation results to JSON and write a readable text summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename_base is None:
        timestamp = time.strftime("%Y%m%d_%H_%M_%S")
        model_name = _sanitize_filename_part(str(results.get("model") or "model"))
        task = _sanitize_filename_part(str(results.get("task") or "task"))
        sample_count = results.get("samples")
        sample_part = str(sample_count) if sample_count is not None else "unknown"
        filename_base = f"{model_name}-{task}-{timestamp}-{sample_part}"
    
    json_path = output_dir / f"{filename_base}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if "classification_metrics" in results:
        try:
            from .classification import format_classification_summary
            summary_text = format_classification_summary(results["classification_metrics"])
            txt_path = output_dir / f"{filename_base}_summary.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(summary_text)
        except Exception as e:
            print(f"Warning: failed to generate text summary: {e}")
    
    return json_path


def _sanitize_filename_part(value: str) -> str:
    """Replace special filename characters with underscores."""
    if not value:
        return "unknown"
    forbidden = '/\\:*?"<>|'
    sanitized = "".join("_" if ch in forbidden else ch for ch in value)
    return sanitized.replace(" ", "_")

