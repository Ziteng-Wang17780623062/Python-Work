import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .settings import PROTOCOL_COLUMNS


def read_protocol(protocol_path: Path) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Read protocol CSV file and extract label descriptions and evaluation scales.
    
    Args:
        protocol_path: Path to the protocol CSV file
        
    Returns:
        Tuple of (label_desc, label_scales) where:
        - label_desc: Dictionary mapping label to description
        - label_scales: Dictionary mapping label to evaluation scales (1-5)
        
    Raises:
        FileNotFoundError: If protocol file does not exist
        ValueError: If protocol file is invalid or empty
    """
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    
    with protocol_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        label_desc: Dict[str, str] = {}
        label_scales: Dict[str, Dict[str, str]] = {}
        
        for row in reader:
            label = row.get("label")
            if not label:
                continue
            label_desc[label] = row.get(PROTOCOL_COLUMNS["5"], "")
            label_scales[label] = {
                score: row.get(column, "")
                for score, column in PROTOCOL_COLUMNS.items()
            }
        
        if not label_scales:
            raise ValueError(f"Protocol file is empty or contains no valid labels: {protocol_path}")
        
        print(f"[协议加载] 从 {protocol_path} 成功加载 {len(label_scales)} 个标签的评估标准")
        return label_desc, label_scales


def load_dataset(dataset_path: Path, limit: Optional[int] = None, random_seed: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    加载数据集，如果指定了 limit，则进行随机采样。

    额外约束（为 LLMs-Mental-Health-Crisis 特别添加）：
    - 当数据集中存在多个标签（如 7 个危机类别）且 limit 充足时，
      采样结果会尽量保证 **包含所有出现过的标签各至少 1 条样本**。
      若在合理重试次数内无法满足，则退化为普通随机采样并给出提示。
    
    Args:
        dataset_path: 数据集文件路径
        limit: 采样数量限制，如果为 None 则返回全部数据
        random_seed: 随机种子，如果为 None 则使用当前时间戳确保每次采样不同
        
    Returns:
        (数据集列表, 使用的随机种子)
        如果没有采样，随机种子为 None
    """
    with dataset_path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # 统计数据集中实际出现的标签
    present_labels = {
        str(item.get("label")).strip()
        for item in data
        if item.get("label") is not None and str(item.get("label")).strip()
    }

    used_seed = None
    if limit is not None and limit < len(data):
        # ===== 决定本次分层采样所依据的标签集合 =====
        # 原本希望使用固定的 7 类标签；但如果数据集中缺少某些标签，就退而求其次，
        # 在“实际存在的标签”上做均匀采样，并给出警告，避免直接报错中断。
        # present_labels 已在上方统计完成。
        # 如果你后续补齐了 7 类标签，这里会自动在 7 类上做均匀采样。
        # 为了兼容性，这里不再硬编码目标标签集合，直接使用 present_labels。
        effective_labels = sorted(present_labels)
        num_labels = len(effective_labels)

        if num_labels == 0:
            raise ValueError("[采样配置错误] 数据集中未检测到任何非空标签，无法进行分层采样。")

        # 设置随机种子，如果未指定则使用时间戳确保每次采样不同
        if random_seed is None:
            # 使用高精度时间戳、进程ID和线程ID组合，确保每次运行都不同
            # 使用 time.perf_counter() 获取更高精度的时间戳（纳秒级）
            timestamp_ns = int(time.perf_counter() * 1000000000)  # 纳秒级时间戳
            process_id = os.getpid()
            # 使用线程ID增加随机性（如果可用）
            try:
                import threading
                thread_id = threading.get_ident()
            except Exception:
                thread_id = 0
            # 使用时间戳的微秒部分增加额外随机性
            time_microseconds = int((time.time() % 1) * 1000000)
            random_seed = (timestamp_ns + process_id * 1000000 + thread_id * 1000 + time_microseconds) % (2**32)

        used_seed = random_seed
        random.seed(random_seed)

        # 如果 limit 小于标签数，无法保证"每类至少 1 条"，改为普通随机采样
        if limit < num_labels:
            print(f"[数据采样警告] 当前数据集中检测到 {num_labels} 个标签，但 limit={limit}，")
            print(f"  无法保证每类至少 1 条样本，将使用普通随机采样（不保证标签分布均匀）。")
            print(f"[数据采样] 使用随机种子: {random_seed}, 从 {len(data)} 条数据中随机采样 {limit} 条")
            # 使用普通随机采样
            sampled = random.sample(data, limit)
            return sampled, used_seed

        print(f"[数据采样] 使用随机种子: {random_seed}, 从 {len(data)} 条数据中分层均匀采样 {limit} 条")

        # ===== 分层均匀随机采样：保证所有出现的标签都至少 1 条，且分布尽可能均衡 =====
        # 1. 按标签分桶
        label_to_items: Dict[str, List[Dict[str, Any]]] = {label: [] for label in effective_labels}
        for item in data:
            label = str(item.get("label")).strip()
            if label in label_to_items:
                label_to_items[label].append(item)

        # 2. 计算每类目标采样数量（尽量均匀分布，差异不超过 1）
        base_per_label = limit // num_labels
        remainder = limit % num_labels

        # 为了避免总是同一批标签多 1 条，对标签顺序做一次随机打乱
        label_list = list(effective_labels)
        random.shuffle(label_list)

        desired_per_label: Dict[str, int] = {}
        total_deficit = 0  # 记录由于某些类别样本不足而“欠下”的配额
        # 首轮分配：尽量按均匀目标分配，不足的类先全取完，记录欠多少
        for idx, label in enumerate(label_list):
            target = base_per_label + (1 if idx < remainder else 0)
            available = len(label_to_items[label])
            take = min(target, available)
            desired_per_label[label] = take
            if available < target:
                # 该类样本不够，无法达到目标数量，把欠下的部分留给其他类别来补
                deficit = target - available
                total_deficit += deficit
                print(
                    f"[采样提示] 标签 '{label}' 只有 {available} 条样本，"
                    f"目标采样数为 {target} 条，将全部采样该标签，并把多余的 {deficit} 条"
                    f"配额分配给其他标签。"
                )

        # 第二轮：将欠下的配额（total_deficit）分配给仍有富余样本的类别
        if total_deficit > 0:
            # 为了随机性，对可用标签顺序再打乱一次
            redistribute_labels = label_list.copy()
            random.shuffle(redistribute_labels)

            for label in redistribute_labels:
                if total_deficit <= 0:
                    break
                available = len(label_to_items[label])
                current = desired_per_label.get(label, 0)
                spare = max(0, available - current)
                if spare <= 0:
                    continue
                add = min(spare, total_deficit)
                desired_per_label[label] = current + add
                total_deficit -= add

            # 从理论上讲，只要 limit <= 总样本数，就应该能完全消化配额；
            # 这里加一个安全检查，如果仍然有欠账，说明实现或数据有异常。
            if total_deficit > 0:
                raise RuntimeError(
                    "[采样内部错误] 在重新分配不足类别的配额时出现异常，"
                    f"仍有 {total_deficit} 条配额无法分配，请检查数据或采样实现。"
                )

        # 3. 在每个标签桶内随机采样对应数量，然后合并并再整体打乱
        sampled: List[Dict[str, Any]] = []
        for label, desired in desired_per_label.items():
            bucket = label_to_items[label]
            # 为避免修改原列表，使用 random.sample
            sampled.extend(random.sample(bucket, desired))

        # 最后整体 shuffle 一下，避免标签按块集中
        random.shuffle(sampled)

        # 安全检查：总数应该等于 limit，且所有有效标签都至少出现一次
        sampled_labels = {
            str(item.get("label")).strip()
            for item in sampled
            if item.get("label") is not None and str(item.get("label")).strip()
        }
        if len(sampled) != limit or not set(effective_labels).issubset(sampled_labels):
            raise RuntimeError(
                "[采样内部错误] 分层采样结果不符合预期，请检查实现。"
                f"len(sampled)={len(sampled)}, 期望={limit}；"
                f"覆盖标签={sorted(list(sampled_labels))}"
            )

        data = sampled
    
    return data, used_seed


def save_results(results: Dict[str, Any], output_dir: Path) -> Path:
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
    
    # 如果包含分类评估结果，同时保存可读的文本摘要
    if "classification_metrics" in results:
        try:
            from .classification import format_classification_summary
            summary_text = format_classification_summary(results["classification_metrics"])
            txt_path = output_dir / f"{filename_base}_summary.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(summary_text)
        except Exception:
            # 如果生成摘要失败，不影响主流程
            pass
    
    
    return json_path


def _sanitize_filename_part(value: str) -> str:
    """将文件名中的特殊字符替换为下划线。"""
    if not value:
        return "unknown"
    forbidden = '/\\:*?"<>|'
    sanitized = "".join("_" if ch in forbidden else ch for ch in value)
    return sanitized.replace(" ", "_")

