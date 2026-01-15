"""
数据集加载和保存工具。
"""

import csv
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .settings import DATASETS, DATA_DIR, USED_DATA_DIR


def _read_csv_with_encoding(dataset_path: Path) -> pd.DataFrame:
    """
    尝试使用多种编码格式读取CSV文件。
    
    Args:
        dataset_path: CSV文件路径
    
    Returns:
        DataFrame对象
    
    Raises:
        UnicodeDecodeError: 如果所有编码都失败
    """
    # 常见的编码格式列表（按优先级排序）
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1", "cp1252"]
    
    for encoding in encodings:
        try:
            df = pd.read_csv(dataset_path, encoding=encoding)
            print(f"成功使用编码 {encoding} 读取文件")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # 如果是其他错误（如解析错误），也尝试下一个编码
            print(f"使用编码 {encoding} 时出现错误: {e}，尝试下一个编码...")
            continue
    
    # 如果所有编码都失败，抛出异常
    raise ValueError(
        f"无法使用常见编码格式读取文件。已尝试的编码: {', '.join(encodings)}"
    )


def load_dataset(
    dataset_name: str,
    limit: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    加载指定数据集。
    
    Args:
        dataset_name: 数据集名称（OEQ, AITA-YTA, PAS）
        limit: 采样数量限制
        random_seed: 随机种子（如果为None，则自动生成）
    
    Returns:
        (samples, random_seed): 样本列表和使用的随机种子
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"未知的数据集: {dataset_name}. 可选: {list(DATASETS.keys())}")
    
    dataset_info = DATASETS[dataset_name]
    dataset_path = dataset_info["path"]
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    # 生成随机种子
    if random_seed is None:
        timestamp = int(time.time() * 1000000)  # 微秒时间戳
        process_id = os.getpid()
        random_seed = (timestamp + process_id) % (2**32)
        print(f"自动生成随机种子: {random_seed}")
    else:
        print(f"使用指定随机种子: {random_seed}")
    
    # 读取CSV文件（自动尝试多种编码）
    print(f"正在加载数据集: {dataset_path}")
    df = _read_csv_with_encoding(dataset_path)
    print(f"原始数据集大小: {len(df)} 行")
    
    # 获取prompt列名
    prompt_column = dataset_info["prompt_column"]
    if prompt_column not in df.columns:
        raise ValueError(f"数据集 {dataset_name} 中未找到列: {prompt_column}")
    
    # 过滤空值
    df = df[df[prompt_column].notna()]
    df = df[df[prompt_column].astype(str).str.strip() != ""]
    print(f"清理后数据集大小: {len(df)} 行")
    
    # 采样
    if limit is not None and limit < len(df):
        random.seed(random_seed)
        df = df.sample(n=limit, random_state=random_seed).reset_index(drop=True)
        print(f"采样后数据集大小: {len(df)} 行 (种子: {random_seed})")
    
    # 转换为样本列表
    samples = []
    for idx, row in df.iterrows():
        sample = {
            "index": idx,
            "prompt": str(row[prompt_column]).strip(),
        }
        # 保存所有原始列
        for col in df.columns:
            sample[col] = row[col]
        samples.append(sample)
    
    return samples, random_seed


def save_sampled_data(
    dataset_name: str,
    samples: List[Dict[str, Any]],
    timestamp: str
) -> Path:
    """
    保存采样数据到 used_data 目录。
    
    Args:
        dataset_name: 数据集名称
        samples: 样本列表
        timestamp: 时间戳字符串
    
    Returns:
        保存的文件路径
    """
    USED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"{dataset_name}-{timestamp}.csv"
    filepath = USED_DATA_DIR / filename
    
    # 将样本转换为DataFrame
    df = pd.DataFrame(samples)
    
    # 保存为CSV
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"采样数据已保存到: {filepath}")
    
    return filepath


def load_sampled_data(filepath: Path) -> List[Dict[str, Any]]:
    """
    从 used_data 目录加载采样数据。
    
    Args:
        filepath: CSV文件路径
    
    Returns:
        样本列表
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    samples = df.to_dict("records")
    return samples


def update_sampled_data_with_responses(
    filepath: Path,
    responses: List[str]
) -> None:
    """
    在采样数据文件中添加被测试LLM的回应列。
    
    Args:
        filepath: CSV文件路径
        responses: 被测试LLM的回应列表（与样本顺序对应）
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    
    if len(responses) != len(df):
        raise ValueError(f"回复数量 ({len(responses)}) 与样本数量 ({len(df)}) 不匹配")
    
    # 添加被测试LLM的回应列
    df["llm_response"] = responses
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"已更新文件，添加了被测试LLM的回应列 (llm_response): {filepath}")


def update_sampled_data_with_judgments(
    filepath: Path,
    judgments: List[Any]
) -> None:
    """
    在采样数据文件中添加judge LLM的判决列（0/1）。
    
    Args:
        filepath: CSV文件路径
        judgments: judge LLM的判决列表（0/1，与样本顺序对应）
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    
    if len(judgments) != len(df):
        raise ValueError(f"裁决数量 ({len(judgments)}) 与样本数量 ({len(df)}) 不匹配")
    
    # 确保judgments是整数类型（0或1）
    judgments_int = [int(j) for j in judgments]
    
    # 添加judge LLM的判决列（0/1）
    df["judge_result"] = judgments_int
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"已更新文件，添加了judge LLM的判决列 (judge_result, 0/1): {filepath}")

