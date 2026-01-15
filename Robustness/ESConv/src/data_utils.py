"""Utilities for loading ESConv data and saving results."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _normalize_label_from_dataset(label: str) -> str:
    """
    Normalize ESConv dataset labels (problem_type) into standard labels.
    ESConv 数据集的标签是 problem_type 字段，包含 12 个问题类型。
    """
    if not label or pd.isna(label):
        # ESConv 默认标签：使用最常见的标签
        return "ongoing depression"
    
    label_str = str(label).strip()
    
    # Lowercase and standardize
    label_str = label_str.lower()
    
    # ESConv 标签映射（标准化格式）
    label_mapping = {
        "ongoing depression": "ongoing depression",
        "breakup with partner": "breakup with partner",
        "job crisis": "job crisis",
        "problems with friends": "problems with friends",
        "academic pressure": "academic pressure",
        "procrastination": "procrastination",
        "alcohol abuse": "alcohol abuse",
        "issues with parent": "issues with parent",
        "sleep problems": "sleep problems",
        "appearance anxiety": "appearance anxiety",
        "school bullying": "school bullying",
        "issues with children": "issues with children",
    }
    
    # If not in mapping, keep lowercase label (with space normalization)
    normalized = label_mapping.get(label_str, label_str)
    # 标准化空格（多个空格变为单个空格）
    normalized = " ".join(normalized.split())
    return normalized


def load_dataset(
    dataset_path: Path, 
    limit: Optional[int] = None, 
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load ESConv dataset (JSON or CSV). Returns normalized samples.
    
    ESConv 数据集支持两种格式：
    1. JSON 格式：包含 problem_type, emotion_type, situation, dialog 等字段
    2. CSV 格式：包含 text 和 label 两列（兼容格式）
    """
    import os
    import time
    import random as random_module
    
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
    
    # 根据文件扩展名判断格式
    file_ext = dataset_path.suffix.lower()
    
    if file_ext == ".json":
        # JSON 格式：ESConv 原始格式
        return _load_json_dataset(dataset_path, limit, random_seed)
    elif file_ext == ".csv":
        # CSV 格式：兼容格式
        return _load_csv_dataset(dataset_path, limit, random_seed)
    else:
        # 尝试自动检测：先尝试 JSON，再尝试 CSV
        try:
            return _load_json_dataset(dataset_path, limit, random_seed)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _load_csv_dataset(dataset_path, limit, random_seed)


def _load_json_dataset(
    dataset_path: Path,
    limit: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load ESConv JSON format dataset."""
    import random as random_module
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Raw dataset size: {len(data)} conversations")
    
    # 过滤无效数据
    valid_data = []
    for item in data:
        if not isinstance(item, dict):
            continue
        
        # ESConv 数据格式检查
        problem_type = item.get("problem_type")
        dialog = item.get("dialog", [])
        
        # 必须有 problem_type 和 dialog
        if not problem_type or not isinstance(problem_type, str) or not problem_type.strip():
            continue
        
        if not isinstance(dialog, list) or len(dialog) == 0:
            continue
        
        # 确保对话不为空（至少有一个有效的对话轮次）
        has_valid_content = False
        for turn in dialog:
            if isinstance(turn, dict):
                content = turn.get("content", "")
                if content and isinstance(content, str) and content.strip():
                    has_valid_content = True
                    break
        
        if not has_valid_content:
            continue
        
        valid_data.append(item)
    
    print(f"Dataset size after cleaning: {len(valid_data)} conversations")
    
    # 采样
    if limit is not None and limit < len(valid_data):
        if random_seed is not None:
            random_module.seed(random_seed)
        valid_data = random_module.sample(valid_data, limit)
        print(f"Dataset size after sampling: {len(valid_data)} conversations (seed: {random_seed})")
    
    samples = []
    for idx, item in enumerate(valid_data):
        problem_type = item.get("problem_type", "")
        situation = item.get("situation", "")
        emotion_type = item.get("emotion_type", "")
        dialog = item.get("dialog", [])
        
        # 构建对话文本：将对话转换为文本格式
        # ESConv 对话格式：每个 turn 包含 speaker (seeker/supporter) 和 content
        dialog_texts = []
        for turn in dialog:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "").strip()
            content = turn.get("content", "").strip()
            
            # 跳过空内容
            if not content:
                continue
            
            # 格式化：speaker: content
            # 注意：ESConv 中 speaker 可能是 "seeker" 或 "supporter"
            if speaker.lower() == "seeker":
                dialog_texts.append(f"Seeker: {content}")
            elif speaker.lower() == "supporter":
                dialog_texts.append(f"Supporter: {content}")
            else:
                # 处理其他可能的 speaker 值
                dialog_texts.append(f"{speaker.capitalize()}: {content}")
        
        # 组合文本：situation + 对话
        # 格式：先显示 situation（如果有），然后显示完整对话
        text_parts = []
        if situation and situation.strip():
            text_parts.append(f"Situation: {situation.strip()}")
        if dialog_texts:
            text_parts.append("\n".join(dialog_texts))
        
        text = "\n\n".join(text_parts) if text_parts else ""
        
        if not text:
            continue
        
        # 标准化标签
        label = _normalize_label_from_dataset(problem_type)
        
        inputs = [text]
        
        sample = {
            "inputs": inputs,
            "label": label,
            "text": text,
            "raw_label": problem_type,
            "index": idx,
            "meta_emotion_type": emotion_type,
            "meta_situation": situation,
            "meta_dialog_length": len(dialog),
            "meta_dialog": dialog,  # 保存原始 dialog 信息，用于提取参考文本
        }
        
        samples.append(sample)
    
    label_counts = {}
    for sample in samples:
        label = sample["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    return samples


def _load_csv_dataset(
    dataset_path: Path,
    limit: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load CSV format dataset (compatibility mode)."""
    import os
    
    df = pd.read_csv(dataset_path, encoding="utf-8")
    
    print(f"Raw dataset size: {len(df)} rows")
    
    text_column = None
    label_column = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["text", "post", "content", "message", "input"]:
            text_column = col
        elif col_lower in ["label", "labels", "class", "category", "problem_type"]:
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
    
    if "generation_metrics" in results:
        try:
            from .classification import format_generation_summary
            summary_text = format_generation_summary(results["generation_metrics"])
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
