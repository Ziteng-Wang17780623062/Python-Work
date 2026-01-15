"""SWMH classification task with streamlined evaluation metrics."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .prompts import build_classification_prompt
from .settings import CLASSIFICATION_MODEL, CLIENT

CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Logger setup
_loggers = {}  # 使用字典存储不同配置的logger


def _setup_logger(
    logs_dir: str = "outputs/logs",
    perturbation_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> logging.Logger:
    """Configure logger and write to both file and console.
    
    Args:
        logs_dir: 日志目录
        perturbation_type: 扰动类型（character/word/sentence/all/none）
        intensity: 扰动强度（none/low/medium/high）
    """
    global _loggers
    
    # 生成日志文件名
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        logs_path = project_root / logs_dir
    logs_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建日志文件名，包含扰动信息
    log_filename_parts = ["classification", timestamp]
    if perturbation_type and perturbation_type != "none":
        log_filename_parts.append(perturbation_type)
    if intensity and intensity != "none":
        log_filename_parts.append(intensity)
    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = logs_path / log_filename
    
    # 使用文件路径作为key，避免重复创建
    log_key = str(log_file)
    if log_key in _loggers:
        return _loggers[log_key]
    
    # 创建新的logger（每次实验都创建新的日志文件）
    logger_name = f"classification_{timestamp}_{perturbation_type or 'none'}_{intensity or 'none'}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 清除旧的handlers（如果有）
    logger.handlers.clear()
    
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    _loggers[log_key] = logger
    logger.info(f"Logger initialized, log file: {log_file}")
    if perturbation_type or intensity:
        logger.info(f"Perturbation: type={perturbation_type or 'none'}, intensity={intensity or 'none'}")
    return logger


def get_logger(
    perturbation_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> logging.Logger:
    """Return logger instance.
    
    Args:
        perturbation_type: 扰动类型（character/word/sentence/all/none）
        intensity: 扰动强度（none/low/medium/high）
    """
    return _setup_logger(perturbation_type=perturbation_type, intensity=intensity)


def run_classification(
    samples: List[Dict[str, Any]],
    dataset_id: str,
    temperature: float = 0.0,
    perturbation_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run classification inference.
    
    Args:
        samples: 样本列表
        dataset_id: 数据集ID
        temperature: 模型温度参数
        perturbation_type: 扰动类型（character/word/sentence/all/none）
        intensity: 扰动强度（none/low/medium/high）
    """
    logger = get_logger(perturbation_type=perturbation_type, intensity=intensity)
    logger.info(
        f"Starting classification - dataset: {dataset_id}, "
        f"samples: {len(samples)}, model: {CLASSIFICATION_MODEL}, temperature: {temperature}"
    )
    if perturbation_type or intensity:
        logger.info(f"Perturbation: type={perturbation_type or 'none'}, intensity={intensity or 'none'}")
    
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
            prompt = build_classification_prompt(conversation_input, sample_dataset_id)
            
            logger.debug(f"Processing sample {idx+1}/{len(samples)} - dataset: {sample_dataset_id}")
            
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
            logger.debug(f"Sample {idx+1} - ground truth: {gt_label}, prediction: {pred_label}")
            
            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "prediction": prediction,
                }
            )
        except Exception as e:
            error_count += 1
            logger.error(f"Error while processing sample {idx+1}: {str(e)}", exc_info=True)
            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "prediction": {"error": str(e)},
                }
            )
    
    logger.info(
        f"Inference completed - total: {len(samples)}, "
        f"succeeded: {len(samples) - error_count}, failed: {error_count}"
    )
    return results


def _compute_confusion_matrix_binary(
    y_true: List[str], y_pred: List[str], pos_label: str, neg_label: str
) -> tuple:
    """Compute binary confusion matrix counts as (TP, FP, FN, TN)."""
    tp = fp = fn = tn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pos_label and pred == pos_label:
            tp += 1
        elif true == pos_label and pred == neg_label:
            fn += 1
        elif true == neg_label and pred == pos_label:
            fp += 1
        elif true == neg_label and pred == neg_label:
            tn += 1
    return tp, fp, fn, tn


def _precision_score_binary(
    y_true: List[str], y_pred: List[str], pos_label: str, neg_label: str, zero_division: float = 0.0
) -> float:
    """Compute binary precision."""
    tp, fp, fn, tn = _compute_confusion_matrix_binary(
        y_true, y_pred, pos_label, neg_label
    )
    if tp + fp == 0:
        return zero_division
    return tp / (tp + fp)


def _recall_score_binary(
    y_true: List[str], y_pred: List[str], pos_label: str, neg_label: str, zero_division: float = 0.0
) -> float:
    """Compute binary recall."""
    tp, fp, fn, tn = _compute_confusion_matrix_binary(
        y_true, y_pred, pos_label, neg_label
    )
    if tp + fn == 0:
        return zero_division
    return tp / (tp + fn)


def _f1_score_binary(
    y_true: List[str], y_pred: List[str], pos_label: str, neg_label: str, zero_division: float = 0.0
) -> float:
    """Compute binary F1."""
    precision = _precision_score_binary(y_true, y_pred, pos_label, neg_label, zero_division)
    recall = _recall_score_binary(y_true, y_pred, pos_label, neg_label, zero_division)
    if precision + recall == 0:
        return zero_division
    return 2 * precision * recall / (precision + recall)


def _f1_score_macro(
    y_true: List[str], y_pred: List[str], zero_division: float = 0.0
) -> float:
    """Compute macro F1 for multiclass."""
    all_labels = set(y_true) | set(y_pred)
    if not all_labels:
        return zero_division
    
    f1_scores = []
    for label in all_labels:
        y_true_binary = ["positive" if l == label else "negative" for l in y_true]
        y_pred_binary = ["positive" if l == label else "negative" for l in y_pred]
        f1 = _f1_score_binary(y_true_binary, y_pred_binary, "positive", "negative", zero_division)
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else zero_division


def _f1_score_micro(
    y_true: List[str], y_pred: List[str], zero_division: float = 0.0
) -> float:
    """Compute micro F1 for multiclass (equals accuracy for single-label)."""
    if len(y_true) == 0:
        return zero_division
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def _classification_report(
    y_true: List[str], y_pred: List[str], zero_division: float = 0.0
) -> Dict[str, Any]:
    """Generate a lightweight classification report similar to sklearn."""
    all_labels = sorted(set(y_true) | set(y_pred))
    report = {}
    
    for label in all_labels:
        y_true_binary = ["positive" if l == label else "negative" for l in y_true]
        y_pred_binary = ["positive" if l == label else "negative" for l in y_pred]
        
        tp, fp, fn, tn = _compute_confusion_matrix_binary(
            y_true_binary, y_pred_binary, "positive", "negative"
        )
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else zero_division
        support = tp + fn
        
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
    
    macro_avg = {
        "precision": sum(r["precision"] for r in report.values()) / len(report) if report else zero_division,
        "recall": sum(r["recall"] for r in report.values()) / len(report) if report else zero_division,
        "f1-score": sum(r["f1-score"] for r in report.values()) / len(report) if report else zero_division,
        "support": sum(r["support"] for r in report.values())
    }
    
    total_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    micro_avg = {
        "precision": total_correct / len(y_true) if len(y_true) > 0 else zero_division,
        "recall": total_correct / len(y_true) if len(y_true) > 0 else zero_division,
        "f1-score": total_correct / len(y_true) if len(y_true) > 0 else zero_division,
        "support": len(y_true)
    }
    
    report["macro avg"] = macro_avg
    report["micro avg"] = micro_avg
    
    return report


def _compute_confusion_matrix_multiclass(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Compute multiclass confusion matrix as per-label tp/fp/fn/tn.
    """
    cm = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in labels}
    
    for true_label, pred_label in zip(y_true, y_pred):
        for label in labels:
            if true_label == label and pred_label == label:
                cm[label]["tp"] += 1
            elif true_label == label and pred_label != label:
                cm[label]["fn"] += 1
            elif true_label != label and pred_label == label:
                cm[label]["fp"] += 1
            else:  # true_label != label and pred_label != label
                cm[label]["tn"] += 1
    
    return cm


def confusion_matrix_from_labels(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> np.ndarray:
    """Wrapper around sklearn.confusion_matrix with fixed label ordering."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path) -> None:
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2 if cm.size > 0 else 0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_macro_precision(
    y_true: List[str], y_pred: List[str], zero_division: float = 0.0
) -> float:
    """Compute macro precision: mean precision over classes."""
    all_labels = sorted(set(y_true) | set(y_pred))
    if not all_labels:
        return zero_division
    
    precision_scores = []
    for label in all_labels:
        y_true_binary = ["positive" if l == label else "negative" for l in y_true]
        y_pred_binary = ["positive" if l == label else "negative" for l in y_pred]
        precision = _precision_score_binary(y_true_binary, y_pred_binary, "positive", "negative", zero_division)
        precision_scores.append(precision)
    
    return sum(precision_scores) / len(precision_scores) if precision_scores else zero_division


def _compute_macro_recall(
    y_true: List[str], y_pred: List[str], zero_division: float = 0.0
) -> float:
    """Compute macro recall: mean recall over classes."""
    all_labels = sorted(set(y_true) | set(y_pred))
    if not all_labels:
        return zero_division
    
    recall_scores = []
    for label in all_labels:
        y_true_binary = ["positive" if l == label else "negative" for l in y_true]
        y_pred_binary = ["positive" if l == label else "negative" for l in y_pred]
        recall = _recall_score_binary(y_true_binary, y_pred_binary, "positive", "negative", zero_division)
        recall_scores.append(recall)
    
    return sum(recall_scores) / len(recall_scores) if recall_scores else zero_division


def _compute_micro_precision(
    y_true: List[str], y_pred: List[str], zero_division: float = 0.0
) -> float:
    """Compute micro precision (same as accuracy for single-label classification)."""
    return _f1_score_micro(y_true, y_pred, zero_division)


def _compute_micro_recall(
    y_true: List[str], y_pred: List[str], zero_division: float = 0.0
) -> float:
    """Compute micro recall (same as accuracy for single-label classification)."""
    return _f1_score_micro(y_true, y_pred, zero_division)


def score_classification(
    classification_results: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    filename_base: Optional[str] = None,
    perturbation_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics required for SWMH:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro-F1
    - Micro-F1
    - Confusion Matrix (saved as image if output_dir/filename_base provided)
    """
    logger = get_logger(perturbation_type=perturbation_type, intensity=intensity)
    logger.info(f"Start scoring - samples: {len(classification_results)}")
    
    y_true = []
    y_pred = []
    per_sample = []

    # Collect labels
    for idx, sample in enumerate(classification_results):
        gt_label = _normalize_label(sample.get("ground_truth"))
        prediction = sample.get("prediction")
        pred_label = None
        if isinstance(prediction, dict):
            pred_label = _normalize_label(prediction.get("label"))
        elif isinstance(prediction, str):
            pred_label = _normalize_label(prediction)
        
        gt_label_str = gt_label if gt_label is not None else "offmychest"
        pred_label_str = pred_label if pred_label is not None else "offmychest"
        
        y_true.append(gt_label_str)
        y_pred.append(pred_label_str)
        
        per_sample.append(
            {
                "index": idx,
                "ground_truth": gt_label_str,
                "prediction": pred_label_str,
                "correct": gt_label_str == pred_label_str,
            }
        )

    # Labels
    all_labels = sorted(set(y_true) | set(y_pred))
    logger.info(f"Classes detected: {all_labels}")
    
    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{len(y_true)})")
    
    # Macro precision / recall / f1
    precision = _compute_macro_precision(y_true, y_pred, zero_division=0.0)
    recall = _compute_macro_recall(y_true, y_pred, zero_division=0.0)
    macro_f1 = _f1_score_macro(y_true, y_pred, zero_division=0.0)
    micro_f1 = _f1_score_micro(y_true, y_pred, zero_division=0.0)
    logger.info(
        f"Precision (Macro): {precision:.4f} | "
        f"Recall (Macro): {recall:.4f} | "
        f"Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}"
    )
    
    # Confusion matrix (counts) for reporting
    confusion_matrix_stats = _compute_confusion_matrix_multiclass(y_true, y_pred, all_labels)
    logger.info("Confusion matrix computed")
    
    # Per-class report
    classification_report = _classification_report(y_true, y_pred, zero_division=0.0)
    logger.info("Per-class report generated")
    
    # Label statistics
    from collections import Counter
    label_counts_true = Counter(y_true)
    label_counts_pred = Counter(y_pred)
    
    label_statistics = {}
    for label in all_labels:
        label_statistics[label] = {
            "true_count": label_counts_true.get(label, 0),
            "pred_count": label_counts_pred.get(label, 0),
        }
    
    # Save confusion matrix heatmap
    cm_image_path = None
    if output_dir is not None and filename_base is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cm_array = confusion_matrix_from_labels(y_true, y_pred, all_labels)
        cm_image_path = output_dir / f"{filename_base}_confusion_matrix.png"
        try:
            _plot_confusion_matrix(cm_array, all_labels, cm_image_path)
            logger.info(f"Confusion matrix image saved to: {cm_image_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix image: {e}", exc_info=True)
    
    result = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "confusion_matrix": confusion_matrix_stats,
        "classification_report": classification_report,
        "label_statistics": label_statistics,
        "per_sample": per_sample,
        "summary": {
            "total_samples": len(classification_results),
            "correct_predictions": correct,
            "incorrect_predictions": len(classification_results) - correct,
            "num_classes": len(all_labels),
            "classes": all_labels,
        },
    }
    
    if cm_image_path:
        result["confusion_matrix_image"] = str(cm_image_path)
    
    logger.info("=" * 60)
    logger.info("Evaluation summary:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (Macro): {precision:.4f}")
    logger.info(f"  Recall (Macro): {recall:.4f}")
    logger.info(f"  Macro-F1: {macro_f1:.4f}")
    logger.info(f"  Micro-F1: {micro_f1:.4f}")
    logger.info("=" * 60)
    
    return result


def format_classification_summary(metrics: Dict[str, Any]) -> str:
    """Format classification metrics into an English-readable summary."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("SWMH Classification Report")
    lines.append("=" * 80)
    lines.append("")

    summary = metrics.get("summary", {})
    lines.append("[Dataset]")
    lines.append(f"  Total samples: {summary.get('total_samples', 0)}")
    lines.append(f"  Correct: {summary.get('correct_predictions', 0)}")
    lines.append(f"  Incorrect: {summary.get('incorrect_predictions', 0)}")
    lines.append(f"  #Classes: {summary.get('num_classes', 0)}")
    lines.append(f"  Classes: {', '.join(summary.get('classes', []))}")
    lines.append("")

    lines.append("[Overall Metrics]")
    lines.append(f"  Accuracy:   {metrics.get('accuracy', 0.0):.4f}")
    lines.append(f"  Precision:  {metrics.get('precision_macro', metrics.get('precision', 0.0)):.4f} (Macro)")
    lines.append(f"  Recall:     {metrics.get('recall_macro', metrics.get('recall', 0.0)):.4f} (Macro)")
    lines.append(f"  Macro-F1:   {metrics.get('macro_f1', metrics.get('f1', 0.0)):.4f}")
    lines.append(f"  Micro-F1:   {metrics.get('micro_f1', 0.0):.4f}")
    lines.append("")

    classification_report = metrics.get("classification_report", {})
    if classification_report:
        lines.append("[Per-Class Metrics]")
        lines.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
        lines.append("-" * 60)
        for label in sorted(classification_report.keys()):
            if label in ["macro avg", "micro avg", "weighted avg"]:
                continue
            report = classification_report[label]
            lines.append(
                f"{label:<15} "
                f"{report.get('precision', 0.0):<10.4f} "
                f"{report.get('recall', 0.0):<10.4f} "
                f"{report.get('f1-score', 0.0):<10.4f} "
                f"{report.get('support', 0):<8}"
            )
        lines.append("")

    confusion_stats = metrics.get("confusion_matrix", {})
    if confusion_stats:
        lines.append("[Confusion Matrix (per-class TP/FP/FN/TN)]")
        lines.append(f"{'Class':<15} {'TP':<8} {'FP':<8} {'FN':<8} {'TN':<8}")
        lines.append("-" * 60)
        for label in sorted(confusion_stats.keys()):
            cm = confusion_stats[label]
            lines.append(
                f"{label:<15} "
                f"{cm.get('tp', 0):<8} "
                f"{cm.get('fp', 0):<8} "
                f"{cm.get('fn', 0):<8} "
                f"{cm.get('tn', 0):<8}"
            )
        lines.append("")

    label_statistics = metrics.get("label_statistics", {})
    if label_statistics:
        lines.append("[Label Distribution]")
        lines.append(f"{'Class':<15} {'Ground Truth':<14} {'Predicted':<12}")
        lines.append("-" * 60)
        for label in sorted(label_statistics.keys()):
            stats = label_statistics[label]
            lines.append(
                f"{label:<15} "
                f"{stats.get('true_count', 0):<14} "
                f"{stats.get('pred_count', 0):<12}"
            )
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def _normalize_label(label: Optional[str]) -> Optional[str]:
    """
    Normalize labels coming from the dataset or model output.
    Valid labels: depression, suicidal, anxiety, offmychest, bipolar.
    """
    if label is None:
        return None
    
    cleaned = str(label).strip()
    
    # Remove "self." prefix if present
    if cleaned.startswith("self."):
        cleaned = cleaned[5:]
    
    cleaned = cleaned.lower()
    
    valid_labels = {"depression", "suicidal", "anxiety", "offmychest", "bipolar"}
    
    # Map common variants
    if cleaned in ["suicidal", "suicide", "suicidal_ideation", "suicidewatch"]:
        return "suicidal"
    if cleaned in ["depression", "depressed", "depressive"]:
        return "depression"
    if cleaned in ["anxiety", "anxious"]:
        return "anxiety"
    if cleaned in ["bipolar", "bipolar_disorder"]:
        return "bipolar"
    if cleaned == "offmychest":
        return "offmychest"
    
    if cleaned in valid_labels:
        return cleaned
    
    if cleaned in ["normal", "none", ""]:
        return "offmychest"
    
    logger = get_logger()
    logger.warning(f"Unknown label '{cleaned}', mapped to 'offmychest'")
    return "offmychest"


def _parse_prediction_content(content: str) -> Dict[str, Any]:
    """Try to extract JSON prediction from model response, supporting Markdown code fences."""
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
            logger.debug(f"JSON parse failed: {str(e)}, candidate preview: {candidate[:100]}...")
            continue

    logger.warning(f"Could not parse JSON, return raw content as label. Preview: {cleaned[:200]}...")
    return {"label": cleaned, "confidence": None, "rationale": None}

