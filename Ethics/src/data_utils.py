"""
数据加载与结果保存工具。

设计原则：
- 按 EthicsMH 数据字段解析，避免假定多余字段。
- 默认基于时间与进程号生成随机种子，保证每次采样不同。
- 保存结果的命名规则参考 C-SSRS。
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .settings import OUTPUT_DIR


def generate_seed() -> int:
    """基于时间戳和进程号生成随机种子，确保每次运行采样不同。"""
    timestamp = int(time.time() * 1_000_000)
    process_id = os.getpid()
    return (timestamp + process_id) % (2**32)


def load_dataset(
    dataset_path: Path,
    sample_size: int,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    加载 EthicsMH 数据集并完成随机采样。

    Args:
        dataset_path: 数据文件路径（CSV 或 Excel）。
        sample_size: 需要采样的条数，必须大于 0。
        random_seed: 随机种子；若为 None 则自动生成，保证每次不同。
    """
    if sample_size <= 0:
        raise ValueError("sample_size 必须为正整数")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    seed = random_seed if random_seed is not None else generate_seed()
    print(f"使用随机种子: {seed}（未指定则自动生成，确保本次采样不同）")

    # 读取数据：支持 CSV 与 Excel；CSV 增加编码回退，避免 UTF-8 失败
    if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(dataset_path)
    else:
        try:
            df = pd.read_csv(dataset_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(dataset_path, encoding="gb18030")

    total = len(df)
    if sample_size > total:
        raise ValueError(f"采样数量 {sample_size} 超过数据集大小 {total}")

    # 随机采样，保证无重复
    sampled = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # 将记录转换为统一的字典列表，避免后续耦合
    records: List[Dict[str, Any]] = []
    for idx, row in sampled.iterrows():
        record = {
            "id": row.get("id", idx),
            "subcategory": row.get("subcategory"),
            "scenario": row.get("scenario"),
            "options": row.get("options"),
            "reasoning_task": row.get("reasoning_task"),
            "expected_reasoning": row.get("expected_reasoning"),
            "model_behavior": row.get("model_behavior"),
            "real_world_impact": row.get("real_world_impact"),
            "viewpoints": row.get("viewpoints"),
            "dataset_id": dataset_path.name,
        }
        records.append(record)

    return records


def save_results(
    results: Dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
    summary_text: Optional[str] = None,
) -> Path:
    """
    参考 C-SSRS 的保存方式：主 JSON + 同名文本摘要（可选）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H_%M_%S")

    model_name = _sanitize_filename_part(str(results.get("model") or "model"))
    task_name = _sanitize_filename_part(str(results.get("task") or "task"))
    sample_count = results.get("samples")
    sample_part = str(sample_count) if sample_count is not None else "unknown"

    filename_base = f"{model_name}-{task_name}-{timestamp}-{sample_part}"
    json_path = output_dir / f"{filename_base}.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 可选：同名文本摘要
    if summary_text is not None:
        txt_path = output_dir / f"{filename_base}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(summary_text)

    return json_path


def _sanitize_filename_part(value: str) -> str:
    """对文件名片段进行清理，防止非法字符。"""
    forbidden = '/\\:*?"<>|'
    sanitized = "".join("_" if ch in forbidden else ch for ch in value)
    return sanitized.replace(" ", "_") or "unknown"


__all__ = ["load_dataset", "save_results", "generate_seed"]
