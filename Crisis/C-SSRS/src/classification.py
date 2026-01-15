"""
C-SSRS 数据集分类任务实现。
实现指标：Macro-F1, Binary F1-score, Precision, Recall, Confusion Matrix
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from .prompts import build_classification_prompt
from .settings import (
    CLASSIFICATION_MODEL,
    CLIENT,
)

CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# 日志配置
_logger = None


def _setup_logger(logs_dir: str = "outputs/logs") -> logging.Logger:
    """配置日志记录器，将日志写入文件和控制台。"""
    global _logger
    if _logger is not None:
        return _logger

    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        logs_path = project_root / logs_dir
    logs_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_path / f"classification_{timestamp}.log"

    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger("classification")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    _logger = logger
    logger.info(f"Logger initialized, log file: {log_file}")
    return logger


def get_logger() -> logging.Logger:
    """获取日志记录器实例。"""
    if _logger is None:
        return _setup_logger()
    return _logger


def run_classification(
    samples: List[Dict[str, Any]],
    dataset_id: str,
    experiment: str = "exp1",
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    运行分类推理任务。
    统一使用5-label分类。
    
    Args:
        samples: 样本列表
        dataset_id: 数据集标识符
        experiment: 保留参数以兼容接口，实际统一使用5-label分类
        temperature: 模型温度参数
        
    Returns:
        分类结果列表
    """
    logger = get_logger()
    logger.info(
        f"Starting classification task - Dataset: {dataset_id}, "
        f"Samples: {len(samples)}, Model: {CLASSIFICATION_MODEL}, Temperature: {temperature}"
    )

    results: List[Dict[str, Any]] = []
    iterator = tqdm(
        samples,
        desc=f"Classification[{dataset_id}]",
        unit="sample",
        leave=False,
        total=len(samples),
    )

    error_count = 0
    for idx, item in enumerate(iterator):
        try:
            conversation_input = item.get("inputs", [])
            sample_dataset_id = item.get("dataset_id") or dataset_id
            prompt = build_classification_prompt(conversation_input, sample_dataset_id, "exp1")

            logger.debug(f"Processing sample {idx+1}/{len(samples)} - Dataset: {sample_dataset_id}")

            completion = CLIENT.chat.completions.create(
                model=CLASSIFICATION_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""

            if not content:
                logger.warning(f"Sample {idx+1} returned empty content")

            prediction = _parse_prediction_content(content)

            pred_label = prediction.get("label") if isinstance(prediction, dict) else str(prediction)
            gt_label = item.get("label")
            logger.debug(f"Sample {idx+1} - True label: {gt_label}, Predicted label: {pred_label}")

            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "prediction": prediction,
                }
            )
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing sample {idx+1}: {str(e)}", exc_info=True)
            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "prediction": {"error": str(e)},
                }
            )

    logger.info(
        f"Classification task completed - Total samples: {len(samples)}, "
        f"Success: {len(samples) - error_count}, Errors: {error_count}"
    )
    return results


def _normalize_label(label: Optional[str], experiment: str = "exp1") -> Optional[str]:
    """
    标准化标签格式。
    使用5-label分类：Supportive、Indicator、Ideation、Behavior、Attempt
    
    Args:
        label: 原始标签
        experiment: 保留参数以兼容接口
        
    Returns:
        标准化后的标签
    """
    if label is None:
        return None

    cleaned = str(label).strip()

    # 标准化标签名称（处理大小写和变体）
    label_mapping = {
        "supportive": "Supportive",
        "indicator": "Indicator",
        "ideation": "Ideation",
        "behavior": "Behavior",
        "attempt": "Attempt",
        # 兼容旧标签
        "no-risk": "Supportive",
        "no_risk": "Supportive",
        "nrisk": "Supportive",
    }

    cleaned_lower = cleaned.lower()
    if cleaned_lower in label_mapping:
        cleaned = label_mapping[cleaned_lower]

    # 使用5-label分类
    valid_labels = {"Supportive", "Indicator", "Ideation", "Behavior", "Attempt"}

    if cleaned in valid_labels:
        return cleaned

    # 如果无法识别，返回原始值（但应该警告）
    logger = get_logger()
    logger.warning(f"Unknown label '{cleaned}', keeping original value")
    return cleaned


def _parse_prediction_content(content: str) -> Dict[str, Any]:
    """从模型响应中提取 JSON 预测结果，兼容 Markdown 代码块。"""
    logger = get_logger()
    cleaned = content.strip()
    candidates = []

    for match in CODE_BLOCK_PATTERN.finditer(cleaned):
        candidates.append(match.group(1))

    candidates.append(cleaned)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed: {str(e)}, candidate content: {candidate[:100]}...")
            continue

    logger.warning(f"Unable to parse as JSON, returning original content as label. Content preview: {cleaned[:200]}...")
    return {"label": cleaned, "confidence": None, "explanation": None}


def _convert_to_binary_labels(labels: List[str], experiment: str = "exp1") -> List[str]:
    """
    将多分类标签转换为二分类标签（有/无自杀意念）。
    使用5-label分类：Supportive和Indicator为无自杀意念，Ideation/Behavior/Attempt为有自杀意念。
    
    Args:
        labels: 原始标签列表
        experiment: 保留参数以兼容接口
        
    Returns:
        二分类标签列表（"With Suicidal Ideation" 或 "Without Suicidal Ideation"）
    """
    binary_labels = []
    # 定义有自杀意念的标签：Ideation, Behavior, Attempt
    # Supportive和Indicator为无自杀意念
    ideation_labels = {"Ideation", "Behavior", "Attempt"}
    
    for label in labels:
        if label in ideation_labels:
            binary_labels.append("With Suicidal Ideation")
        else:
            binary_labels.append("Without Suicidal Ideation")
    
    return binary_labels


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    计算混淆矩阵（字典格式）。
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        labels: 所有类别标签列表
        
    Returns:
        混淆矩阵字典，格式为 {true_label: {pred_label: count}}
    """
    # 计算混淆矩阵字典
    cm_dict = {label: {p: 0 for p in labels} for label in labels}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in cm_dict and pred_label in cm_dict[true_label]:
            cm_dict[true_label][pred_label] += 1

    return cm_dict


def score_classification(
    classification_results: List[Dict[str, Any]],
    experiment: str = "exp1",
    annotator_labels: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """
    计算评估指标：Macro-F1, Binary F1-score, Precision, Recall, Confusion Matrix。
    统一使用5-label分类。
    
    Args:
        classification_results: 分类结果列表
        experiment: 保留参数以兼容接口，实际统一使用5-label分类
        annotator_labels: 多个标注者的标签列表（可选，保留参数以兼容接口）
        
    Returns:
        包含所有评估指标的字典
    """
    logger = get_logger()
    logger.info(f"Starting score calculation - Results count: {len(classification_results)}")

    y_true = []
    y_pred = []
    per_sample = []

    # 收集真实标签和预测标签
    for idx, sample in enumerate(classification_results):
        gt_label_raw = sample.get("ground_truth")
        prediction = sample.get("prediction")

        # 标准化标签
        if isinstance(prediction, dict):
            pred_label_raw = prediction.get("label")
        elif isinstance(prediction, str):
            pred_label_raw = prediction
        else:
            pred_label_raw = None

        gt_label = _normalize_label(gt_label_raw, experiment)
        pred_label = _normalize_label(pred_label_raw, experiment)

        if gt_label is None:
            logger.warning(f"Sample {idx} has empty true label, skipping")
            continue

        if pred_label is None:
            # 如果预测标签为空，使用最低风险标签
            pred_label = "Supportive"

        y_true.append(gt_label)
        y_pred.append(pred_label)

        per_sample.append(
            {
                "index": idx,
                "ground_truth": gt_label,
                "prediction": pred_label,
                "correct": gt_label == pred_label,
            }
        )

    # 获取所有类别，只保留5个有效标签
    valid_labels = {"Supportive", "Indicator", "Ideation", "Behavior", "Attempt"}
    
    # 过滤无效标签
    filtered_y_true = []
    filtered_y_pred = []
    filtered_per_sample = []
    
    for i, (true_lbl, pred_lbl, sample) in enumerate(zip(y_true, y_pred, per_sample)):
        if true_lbl in valid_labels and pred_lbl in valid_labels:
            filtered_y_true.append(true_lbl)
            filtered_y_pred.append(pred_lbl)
            filtered_per_sample.append(sample)
        else:
            logger.warning(f"Sample {i} has invalid labels (True: {true_lbl}, Pred: {pred_lbl}), skipping")
    
    y_true = filtered_y_true
    y_pred = filtered_y_pred
    per_sample = filtered_per_sample
    
    # 按照严重程度排序：Supportive < Indicator < Ideation < Behavior < Attempt
    label_order = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]
    all_labels = [label for label in label_order if label in set(y_true) | set(y_pred)]
    
    # 如果发现无效标签，记录警告
    invalid_labels = set(y_true) | set(y_pred) - valid_labels
    if invalid_labels:
        logger.warning(f"Found invalid labels (will be filtered): {invalid_labels}")
    
    if not all_labels:
        logger.error("No valid labels found after filtering!")
        all_labels = label_order  # 使用默认标签列表
    
    logger.info(f"Dataset contains classes: {all_labels}")

    # 计算 Macro-F1
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    logger.info(f"Macro-F1: {macro_f1:.4f}")

    # 计算 Binary F1-score（有/无自杀意念）
    y_true_binary = _convert_to_binary_labels(y_true, experiment)
    y_pred_binary = _convert_to_binary_labels(y_pred, experiment)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label='With Suicidal Ideation', zero_division=0)
    logger.info(f"Binary F1-score (With/Without Suicidal Ideation): {binary_f1:.4f}")

    # 计算精确率 (Precision) 和召回率 (Recall)
    # Macro 平均
    macro_precision = precision_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    logger.info(f"Macro Precision: {macro_precision:.4f}")
    logger.info(f"Macro Recall: {macro_recall:.4f}")

    # Per-class Precision 和 Recall
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )
    precision_dict = {label: float(prec) for label, prec in zip(all_labels, precision_per_class)}
    recall_dict = {label: float(rec) for label, rec in zip(all_labels, recall_per_class)}

    logger.info("Per-class Precision:")
    for label, prec in precision_dict.items():
        logger.info(f"  {label}: {prec:.4f}")
    logger.info("Per-class Recall:")
    for label, rec in recall_dict.items():
        logger.info(f"  {label}: {rec:.4f}")

    # 计算混淆矩阵（图片将在保存结果时通过可视化模块生成）
    confusion_matrix = compute_confusion_matrix(
        y_true, y_pred, all_labels
    )
    logger.info("Confusion matrix computed (image will be generated when saving results)")

    # 汇总结果
    result = {
        "macro_f1": float(macro_f1),
        "binary_f1": float(binary_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "precision_per_class": precision_dict,
        "recall_per_class": recall_dict,
        "confusion_matrix": confusion_matrix,
        "per_sample": per_sample,
        "summary": {
            "total_samples": len(classification_results),
            "num_classes": len(all_labels),
            "classes": all_labels,
        }
    }

    logger.info("=" * 60)
    logger.info("Evaluation Results Summary:")
    logger.info(f"  Macro-F1: {macro_f1:.4f}")
    logger.info(f"  Binary F1-score (With/Without Suicidal Ideation): {binary_f1:.4f}")
    logger.info(f"  Macro Precision: {macro_precision:.4f}")
    logger.info(f"  Macro Recall: {macro_recall:.4f}")
    logger.info("=" * 60)

    return result


def format_metrics_summary(metrics: Dict[str, Any], experiment: str = "exp1") -> str:
    """
    将评估结果格式化为易读的文本摘要。
    统一使用5-label分类。
    """
    lines = []
    lines.append("=" * 80)
    lines.append("C-SSRS Classification Task Evaluation Report")
    lines.append("=" * 80)
    lines.append("")

    summary = metrics.get("summary", {})
    lines.append("[Sample Statistics]")
    lines.append(f"  Total Samples: {summary.get('total_samples', 0)}")
    lines.append(f"  Number of Classes: {summary.get('num_classes', 0)}")
    lines.append(f"  Class List: {', '.join(summary.get('classes', []))}")
    lines.append("")

    # 评估指标
    lines.append("[Evaluation Metrics]")
    lines.append(f"  Macro-F1: {metrics.get('macro_f1', 0.0):.4f}")
    lines.append(f"  Binary F1-score (With/Without Suicidal Ideation): {metrics.get('binary_f1', 0.0):.4f}")
    lines.append(f"  Macro Precision: {metrics.get('macro_precision', 0.0):.4f}")
    lines.append(f"  Macro Recall: {metrics.get('macro_recall', 0.0):.4f}")
    lines.append("")

    # Per-class Precision 和 Recall
    precision_per_class = metrics.get("precision_per_class", {})
    recall_per_class = metrics.get("recall_per_class", {})
    if precision_per_class or recall_per_class:
        lines.append("[Per-class Precision]")
        for label, prec in sorted(precision_per_class.items()):
            lines.append(f"  {label}: {prec:.4f}")
        lines.append("")
        lines.append("[Per-class Recall]")
        for label, rec in sorted(recall_per_class.items()):
            lines.append(f"  {label}: {rec:.4f}")
        lines.append("")

    # 混淆矩阵（文本形式）
    confusion_matrix = metrics.get("confusion_matrix", {})
    if confusion_matrix:
        lines.append("[Confusion Matrix (Text Format, see corresponding PNG file for visualization)]")
        labels = sorted(set(confusion_matrix.keys()) | set(confusion_matrix[list(confusion_matrix.keys())[0]].keys()))
        
        # 表头
        header_label = "True\\Pred"
        header = f"{header_label:<15}"
        for pred_label in labels:
            header += f"{pred_label:<15}"
        lines.append(header)
        lines.append("-" * (15 * (len(labels) + 1)))
        
        # 表格内容
        for true_label in labels:
            row = f"{true_label:<15}"
            for pred_label in labels:
                count = confusion_matrix.get(true_label, {}).get(pred_label, 0)
                row += f"{count:<15}"
            lines.append(row)
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)

