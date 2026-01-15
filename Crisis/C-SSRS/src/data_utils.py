"""
C-SSRS 数据集加载和结果保存工具。
"""

import ast
import html
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_dataset(
    dataset_path: Path,
    limit: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    加载 C-SSRS 数据集（CSV 格式）。
    
    Args:
        dataset_path: CSV 文件路径
        limit: 采样数量限制，如果为 None 则返回全部数据
        random_seed: 随机种子，如果为 None 则自动生成基于时间的随机种子
        
    Returns:
        转换后的样本列表，格式为 [{"inputs": [...], "label": "...", ...}, ...]
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    # 如果未指定随机种子，自动生成一个基于时间的随机种子
    if random_seed is None:
        timestamp = int(time.time() * 1000000)  # 微秒级时间戳
        process_id = os.getpid()
        random_seed = (timestamp + process_id) % (2**32)
        print(f"Auto-generated random seed: {random_seed}")
    else:
        print(f"Using specified random seed: {random_seed}")
    
    # 读取 CSV 文件
    df = pd.read_csv(dataset_path, encoding="utf-8")
    
    print(f"Original dataset size: {len(df)} samples")
    
    # 数据清理：移除无效数据
    # 1. 移除空Post
    df = df[df["Post"].notna()]
    df = df[df["Post"].astype(str).str.strip() != ""]
    
    # 2. 移除空Label
    df = df[df["Label"].notna()]
    df = df[df["Label"].astype(str).str.strip() != ""]
    
    print(f"Dataset size after cleaning: {len(df)} samples")
    
    # 采样
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=random_seed).reset_index(drop=True)
        print(f"Dataset size after sampling: {len(df)} samples (random seed: {random_seed})")
    
    # 转换为统一格式
    samples = []
    for idx, row in df.iterrows():
        try:
            # 解析Post列（列表格式字符串）
            post_str = str(row["Post"]).strip()
            try:
                # 使用ast.literal_eval安全解析
                post_list = ast.literal_eval(post_str)
                if not isinstance(post_list, list):
                    post_list = [post_str]
            except (ValueError, SyntaxError):
                # 如果解析失败，将整个字符串作为单个帖子
                post_list = [post_str]
            
            # HTML实体解码
            post_list = [html.unescape(str(post)) for post in post_list]
            
            # 合并所有帖子为单个文本
            post_text = " ".join(post_list)
            
            # 标准化标签
            label = str(row["Label"]).strip()
            
            # 将文本转换为列表格式（兼容其他项目的格式）
            inputs = [post_text]
            
            sample = {
                "inputs": inputs,
                "label": label,
                "post_text": post_text,
                "post_list": post_list,
                "user": str(row.get("User", f"user-{idx}")),
                "index": idx,
            }
            
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Skipping sample {idx}, error: {e}")
            continue
    
    # 统计标签分布
    label_counts = {}
    for sample in samples:
        label = sample["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    return samples


def save_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """
    保存评估结果到 JSON 文件，并生成可读的文本摘要和混淆矩阵图片。
    
    Args:
        results: 评估结果字典
        output_dir: 输出目录
        
    Returns:
        保存的 JSON 文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H_%M_%S")
    model_name = _sanitize_filename_part(str(results.get("model") or "model"))
    task = _sanitize_filename_part(str(results.get("task") or "task"))
    sample_count = results.get("samples")
    sample_part = str(sample_count) if sample_count is not None else "unknown"
    filename_base = f"{model_name}-{task}-{timestamp}-{sample_part}"
    
    # 保存 JSON 文件
    json_path = output_dir / f"{filename_base}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 如果包含评估结果，同时保存可读的文本摘要
    if "metrics" in results:
        try:
            from .classification import format_metrics_summary
            
            # 生成文本摘要
            summary_text = format_metrics_summary(results["metrics"])
            txt_path = output_dir / f"{filename_base}_summary.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(summary_text)
        except Exception as e:
            # 如果生成摘要失败，不影响主流程
            print(f"Warning: Failed to generate text summary: {e}")
            import traceback
            traceback.print_exc()
    
    return json_path


def _sanitize_filename_part(value: str) -> str:
    """将文件名中的特殊字符替换为下划线。"""
    if not value:
        return "unknown"
    forbidden = '/\\:*?"<>|'
    sanitized = "".join("_" if ch in forbidden else ch for ch in value)
    return sanitized.replace(" ", "_")

