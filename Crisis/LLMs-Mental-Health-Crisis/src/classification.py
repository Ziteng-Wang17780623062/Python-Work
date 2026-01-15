import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .prompts import build_classification_prompt
from .settings import CLASSIFICATION_MODEL, CLIENT


CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# 日志配置
_logger = None


def _setup_logger(logs_dir: str = "outputs/logs") -> logging.Logger:
    """配置日志记录器，将日志写入文件和控制台。"""
    global _logger
    if _logger is not None:
        return _logger
    
    # 确保日志目录存在（相对于项目根目录）
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        # 如果相对路径，从当前文件位置向上找到项目根目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # src -> 项目根目录
        logs_path = project_root / logs_dir
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_path / f"classification_{timestamp}.log"
    
    # 配置日志格式
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 创建日志记录器
    logger = logging.getLogger("classification")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    _logger = logger
    logger.info(f"日志记录器已初始化，日志文件: {log_file}")
    return logger


def get_logger() -> logging.Logger:
    """获取日志记录器实例。"""
    if _logger is None:
        return _setup_logger()
    return _logger


def run_classification(
    samples: List[Dict[str, Any]],
    dataset_id: str,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    logger = get_logger()
    logger.info(f"开始分类推理任务 - 数据集: {dataset_id}, 样本数: {len(samples)}, 模型: {CLASSIFICATION_MODEL}, 温度: {temperature}")
    
    results: List[Dict[str, Any]] = []
    iterator = tqdm(
        samples,
        desc=f"分类推理[{dataset_id}]",
        unit="样本",
        leave=False,
        total=len(samples),
    )
    
    error_count = 0
    for idx, item in enumerate(iterator):
        try:
            conversation_input = item.get("inputs", [])
            sample_dataset_id = item.get("dataset_id") or dataset_id
            prompt = build_classification_prompt(conversation_input, sample_dataset_id)
            
            logger.debug(f"处理样本 {idx+1}/{len(samples)} - 数据集: {sample_dataset_id}")
            
            completion = CLIENT.chat.completions.create(
                model=CLASSIFICATION_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            
            if not content:
                logger.warning(f"样本 {idx+1} 返回空内容")
            
            prediction = _parse_prediction_content(content)
            
            # 记录预测结果
            pred_label = prediction.get("label") if isinstance(prediction, dict) else str(prediction)
            gt_label = item.get("label")
            logger.debug(f"样本 {idx+1} - 真实标签: {gt_label}, 预测标签: {pred_label}")
            
            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "prediction": prediction,
                }
            )
        except Exception as e:
            error_count += 1
            logger.error(f"处理样本 {idx+1} 时发生错误: {str(e)}", exc_info=True)
            # 即使出错也添加一个结果，标记为错误
            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "prediction": {"error": str(e)},
                }
            )
    
    logger.info(f"分类推理任务完成 - 总样本数: {len(samples)}, 成功: {len(samples) - error_count}, 错误: {error_count}")
    return results


def _is_crisis_label(label: Optional[str]) -> bool:
    """判断标签是否为'有问题'类别。"""
    if label is None:
        return False
    label_str = str(label).strip().lower()
    # "no_crisis" 或空字符串视为"没问题"
    if label_str in ("", "no_crisis", "none", "null"):
        return False
    # 其他所有标签都视为"有问题"
    return True


def _compute_confusion_matrix_binary(y_true: List[str], y_pred: List[str], pos_label: str, neg_label: str) -> tuple:
    """计算二分类混淆矩阵，返回 (TP, FP, FN, TN)"""
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


def _precision_score_binary(y_true: List[str], y_pred: List[str], pos_label: str, zero_division: float = 0.0) -> float:
    """计算二分类精确率"""
    tp, fp, fn, tn = _compute_confusion_matrix_binary(
        y_true, y_pred, pos_label, 
        neg_label="no_crisis" if pos_label == "crisis" else "crisis"
    )
    if tp + fp == 0:
        return zero_division
    return tp / (tp + fp)


def _recall_score_binary(y_true: List[str], y_pred: List[str], pos_label: str, zero_division: float = 0.0) -> float:
    """计算二分类召回率"""
    tp, fp, fn, tn = _compute_confusion_matrix_binary(
        y_true, y_pred, pos_label,
        neg_label="no_crisis" if pos_label == "crisis" else "crisis"
    )
    if tp + fn == 0:
        return zero_division
    return tp / (tp + fn)


def _f1_score_binary(y_true: List[str], y_pred: List[str], pos_label: str, zero_division: float = 0.0) -> float:
    """计算二分类F1分数"""
    precision = _precision_score_binary(y_true, y_pred, pos_label, zero_division)
    recall = _recall_score_binary(y_true, y_pred, pos_label, zero_division)
    if precision + recall == 0:
        return zero_division
    return 2 * precision * recall / (precision + recall)


def _f1_score_macro(y_true: List[str], y_pred: List[str], zero_division: float = 0.0) -> float:
    """计算宏观F1分数（多分类）"""
    # 获取所有类别
    all_labels = set(y_true) | set(y_pred)
    if not all_labels:
        return zero_division
    
    f1_scores = []
    for label in all_labels:
        # 为每个类别计算二分类F1
        y_true_binary = ["positive" if l == label else "negative" for l in y_true]
        y_pred_binary = ["positive" if l == label else "negative" for l in y_pred]
        f1 = _f1_score_binary(y_true_binary, y_pred_binary, "positive", zero_division)
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else zero_division


def _f1_score_micro(y_true: List[str], y_pred: List[str], zero_division: float = 0.0) -> float:
    """计算微观F1分数（多分类，等于整体准确率）"""
    if len(y_true) == 0:
        return zero_division
    
    # 微观F1等于整体准确率
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def _classification_report(y_true: List[str], y_pred: List[str], zero_division: float = 0.0) -> Dict[str, Any]:
    """生成分类报告（类似sklearn的classification_report）"""
    all_labels = sorted(set(y_true) | set(y_pred))
    report = {}
    
    # 为每个类别计算指标
    for label in all_labels:
        y_true_binary = ["positive" if l == label else "negative" for l in y_true]
        y_pred_binary = ["positive" if l == label else "negative" for l in y_pred]
        
        tp, fp, fn, tn = _compute_confusion_matrix_binary(y_true_binary, y_pred_binary, "positive", "negative")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else zero_division
        support = tp + fn  # 真实标签中该类的数量
        
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
    
    # 计算宏平均
    macro_avg = {
        "precision": sum(r["precision"] for r in report.values()) / len(report) if report else zero_division,
        "recall": sum(r["recall"] for r in report.values()) / len(report) if report else zero_division,
        "f1-score": sum(r["f1-score"] for r in report.values()) / len(report) if report else zero_division,
        "support": sum(r["support"] for r in report.values())
    }
    
    # 计算微平均（等于整体准确率）
    total_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    micro_avg = {
        "precision": total_correct / len(y_true) if len(y_true) > 0 else zero_division,
        "recall": total_correct / len(y_true) if len(y_true) > 0 else zero_division,
        "f1-score": total_correct / len(y_true) if len(y_true) > 0 else zero_division,
        "support": len(y_true)
    }
    
    report["macro avg"] = macro_avg
    report["micro avg"] = micro_avg
    report["weighted avg"] = macro_avg  # 简化处理，使用宏平均
    
    return report


def _compute_multiclass_confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    """计算多分类混淆矩阵，返回字典格式 {真实标签: {预测标签: 数量}}"""
    all_labels = sorted(set(y_true) | set(y_pred))
    confusion_matrix = {label: {pred_label: 0 for pred_label in all_labels} for label in all_labels}
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in confusion_matrix:
            confusion_matrix[true_label] = {pred_label: 0 for pred_label in all_labels}
        if pred_label not in confusion_matrix[true_label]:
            confusion_matrix[true_label][pred_label] = 0
        confusion_matrix[true_label][pred_label] += 1
    
    return confusion_matrix


def score_classification(
    classification_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    标准分类评估指标：
    - Macro-F1: 所有类别F1分数的平均值
    - F1(Per class): 每个类别的F1分数
    - 精确率 (Precision)、召回率 (Recall): 每个类别的精确率和召回率
    - 混淆矩阵 (Confusion Matrix): 多分类混淆矩阵
    """
    logger = get_logger()
    logger.info(f"开始评分计算 - 结果数量: {len(classification_results)}")
    
    # 收集真实标签和预测标签（原始标签）
    y_true_full = []
    y_pred_full = []
    per_sample = []

    for idx, sample in enumerate(classification_results):
        gt_label = _normalize_label(sample.get("ground_truth"))
        prediction = sample.get("prediction")
        pred_label = None
        pred_raw = None  # 原始预测内容
        
        if isinstance(prediction, dict):
            pred_label = _normalize_label(prediction.get("label"))
            # 保存完整的原始预测字典
            pred_raw = prediction.copy() if prediction else None
        elif isinstance(prediction, str):
            pred_label = _normalize_label(prediction)
            pred_raw = prediction
        else:
            # 如果prediction是其他类型，转换为字符串
            pred_raw = str(prediction) if prediction is not None else None
        
        # 处理None值，统一为字符串
        gt_label_str = gt_label if gt_label is not None else "no_crisis"
        pred_label_str = pred_label if pred_label is not None else "no_crisis"
        
        y_true_full.append(gt_label_str)
        y_pred_full.append(pred_label_str)
        
        # 记录每个回复的详细信息
        per_sample.append(
            {
                "index": idx,
                "ground_truth": gt_label_str,
                "prediction": pred_label_str,
                "prediction_raw": pred_raw,  # 原始预测内容（字典或字符串）
                "inputs": sample.get("inputs"),  # 输入内容
                "correct": gt_label_str == pred_label_str,
            }
        )
        logger.debug(f"样本 {idx+1} - 真实标签: {gt_label_str}, 预测标签: {pred_label_str}, 正确: {gt_label_str == pred_label_str}")

    # ========== 计算多分类指标 ==========
    # 获取所有类别
    all_labels = sorted(set(y_true_full) | set(y_pred_full))
    logger.info(f"检测到的类别: {all_labels}")
    
    # 计算多分类混淆矩阵
    confusion_matrix = _compute_multiclass_confusion_matrix(y_true_full, y_pred_full)
    
    # 计算每个类别的指标
    per_class_metrics = {}
    for label in all_labels:
        # 为每个类别计算二分类指标
        y_true_binary = ["positive" if l == label else "negative" for l in y_true_full]
        y_pred_binary = ["positive" if l == label else "negative" for l in y_pred_full]
        
        tp, fp, fn, tn = _compute_confusion_matrix_binary(y_true_binary, y_pred_binary, "positive", "negative")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn  # 真实标签中该类的数量
        
        per_class_metrics[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        
        logger.info(f"类别 '{label}': Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
    
    # 计算Macro-F1（所有类别F1分数的平均值）
    macro_f1 = sum(metrics["f1"] for metrics in per_class_metrics.values()) / len(per_class_metrics) if per_class_metrics else 0.0
    
    # 计算Macro-Precision和Macro-Recall
    macro_precision = sum(metrics["precision"] for metrics in per_class_metrics.values()) / len(per_class_metrics) if per_class_metrics else 0.0
    macro_recall = sum(metrics["recall"] for metrics in per_class_metrics.values()) / len(per_class_metrics) if per_class_metrics else 0.0
    
    # 计算Micro-F1（等于整体准确率）
    total_correct = sum(1 for t, p in zip(y_true_full, y_pred_full) if t == p)
    micro_f1 = total_correct / len(y_true_full) if len(y_true_full) > 0 else 0.0
    
    logger.info("=" * 60)
    logger.info("评估结果汇总:")
    logger.info(f"  Macro-F1: {macro_f1:.4f}")
    logger.info(f"  Macro-Precision: {macro_precision:.4f}")
    logger.info(f"  Macro-Recall: {macro_recall:.4f}")
    logger.info(f"  Micro-F1 (准确率): {micro_f1:.4f}")
    logger.info(f"  总样本数: {len(classification_results)}")
    logger.info(f"  正确预测数: {total_correct}")
    logger.info("=" * 60)
    
    # ========== 汇总结果 ==========
    result = {
        # 总体指标
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "micro_f1": float(micro_f1),
        "accuracy": float(micro_f1),  # 准确率等于Micro-F1
        
        # 每个类别的指标
        "per_class_metrics": per_class_metrics,
        
        # 混淆矩阵
        "confusion_matrix": confusion_matrix,
        
        # 详细数据
        "per_sample": per_sample,
        
        # 样本统计
        "summary": {
            "total_samples": len(classification_results),
            "total_correct": int(total_correct),
            "total_incorrect": len(classification_results) - int(total_correct),
            "classes": all_labels,
            "class_count": len(all_labels),
        }
    }
    
    return result


def format_classification_summary(metrics: Dict[str, Any]) -> str:
    """
    将分类评估结果格式化为易读的文本摘要。
    
    Args:
        metrics: score_classification 返回的结果字典
        
    Returns:
        格式化的文本摘要
    """
    lines = []
    lines.append("=" * 80)
    lines.append("分类任务评估报告")
    lines.append("=" * 80)
    lines.append("")
    
    # 样本统计
    summary = metrics.get("summary", {})
    lines.append("【样本统计】")
    lines.append(f"  总样本数: {summary.get('total_samples', 0)}")
    lines.append(f"  正确预测数: {summary.get('total_correct', 0)}")
    lines.append(f"  错误预测数: {summary.get('total_incorrect', 0)}")
    lines.append(f"  类别数: {summary.get('class_count', 0)}")
    lines.append(f"  类别列表: {', '.join(summary.get('classes', []))}")
    lines.append("")
    
    # 总体指标
    lines.append("【总体指标】")
    lines.append(f"  Macro-F1: {metrics.get('macro_f1', 0.0):.4f}")
    lines.append(f"  Macro-Precision: {metrics.get('macro_precision', 0.0):.4f}")
    lines.append(f"  Macro-Recall: {metrics.get('macro_recall', 0.0):.4f}")
    lines.append(f"  Micro-F1 (准确率): {metrics.get('micro_f1', 0.0):.4f}")
    lines.append("")
    
    # 每个类别的指标
    per_class_metrics = metrics.get("per_class_metrics", {})
    if per_class_metrics:
        lines.append("【每个类别的指标 (Per Class Metrics)】")
        lines.append("")
        # 表头
        lines.append(f"{'类别':<20} {'精确率(Precision)':<18} {'召回率(Recall)':<18} {'F1分数':<12} {'支持数(Support)':<15}")
        lines.append("-" * 80)
        
        # 按类别排序
        sorted_classes = sorted(per_class_metrics.keys())
        for label in sorted_classes:
            class_metrics = per_class_metrics[label]
            precision = class_metrics.get("precision", 0.0)
            recall = class_metrics.get("recall", 0.0)
            f1 = class_metrics.get("f1", 0.0)
            support = class_metrics.get("support", 0)
            lines.append(f"{label:<20} {precision:<18.4f} {recall:<18.4f} {f1:<12.4f} {support:<15}")
        
        lines.append("")
    
    # 混淆矩阵
    confusion_matrix = metrics.get("confusion_matrix", {})
    if confusion_matrix:
        lines.append("【混淆矩阵 (Confusion Matrix)】")
        lines.append("")
        lines.append("说明: 行表示真实标签，列表示预测标签")
        lines.append("")
        
        # 获取所有标签并排序
        all_labels = sorted(set(confusion_matrix.keys()) | set(
            pred_label for row in confusion_matrix.values() for pred_label in row.keys()
        ))
        
        # 打印表头
        header_label = "真实\\预测"  # 先定义包含反斜杠的字符串
        header = f"{header_label:<15}"
        for pred_label in all_labels:
            header += f"{pred_label:<12}"
        lines.append(header)
        lines.append("-" * (15 + 12 * len(all_labels)))
        
        # 打印每一行
        for true_label in all_labels:
            row_str = f"{true_label:<15}"
            if true_label in confusion_matrix:
                for pred_label in all_labels:
                    count = confusion_matrix[true_label].get(pred_label, 0)
                    row_str += f"{count:<12}"
            else:
                row_str += "0" * 12 * len(all_labels)
            lines.append(row_str)
        
        lines.append("")
        
        # 错误分析：找出最常见的错误类型
        lines.append("【错误分析 (Error Analysis)】")
        error_pairs = []
        for true_label in all_labels:
            if true_label in confusion_matrix:
                for pred_label in all_labels:
                    if true_label != pred_label:
                        count = confusion_matrix[true_label].get(pred_label, 0)
                        if count > 0:
                            error_pairs.append((true_label, pred_label, count))
        
        if error_pairs:
            # 按错误数量排序
            error_pairs.sort(key=lambda x: x[2], reverse=True)
            lines.append("最常见的错误类型（真实标签 -> 预测标签）:")
            for true_label, pred_label, count in error_pairs[:10]:  # 只显示前10个
                lines.append(f"  {true_label} -> {pred_label}: {count} 次")
        else:
            lines.append("  无错误预测")
        
        lines.append("")
    
    # 结果解读
    lines.append("【结果解读】")
    macro_f1 = metrics.get('macro_f1', 0.0)
    micro_f1 = metrics.get('micro_f1', 0.0)
    macro_precision = metrics.get('macro_precision', 0.0)
    macro_recall = metrics.get('macro_recall', 0.0)
    
    if macro_f1 < 0.5:
        lines.append("  ⚠️  Macro-F1较低：模型在各类别上的平均表现不佳。")
    elif macro_f1 >= 0.7:
        lines.append("  ✓  Macro-F1良好：模型在各类别上的平均表现较好。")
    else:
        lines.append("  ⚠️  Macro-F1中等：模型表现有改进空间。")
    
    if abs(macro_precision - macro_recall) > 0.2:
        if macro_precision > macro_recall:
            lines.append("  ⚠️  精确率高于召回率：模型倾向于保守预测，可能漏掉一些正例。")
        else:
            lines.append("  ⚠️  召回率高于精确率：模型倾向于积极预测，可能产生较多误报。")
    else:
        lines.append("  ✓  精确率和召回率较为均衡。")
    
    if abs(macro_f1 - micro_f1) > 0.2:
        lines.append("  ⚠️  Macro-F1与Micro-F1差异较大：模型在不同类别上的表现不均衡，可能在样本量少的类别上表现较差。")
    elif macro_f1 > 0.5 and micro_f1 > 0.5:
        lines.append("  ✓  模型在各个类别上的表现较为均衡。")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    cleaned = str(label).strip()
    return cleaned if cleaned else None


def _parse_prediction_content(content: str) -> Dict[str, Any]:
    """尽量从模型响应中提取 JSON 预测结果，兼容 Markdown 代码块。"""
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
            logger.debug(f"JSON 解析失败: {str(e)}, 候选内容: {candidate[:100]}...")
            continue

    logger.warning(f"无法解析为 JSON，返回原始内容作为标签。内容预览: {cleaned[:200]}...")
    return {"label": cleaned, "confidence": None, "rationale": None}



