"""ESConv 生成任务评估模块。

实现对话生成、指标计算等功能。
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import nltk
from tqdm import tqdm

from .metrics import Metric, ensure_nltk
from .prompts import build_generation_prompt, format_turn
from .settings import CLIENT, GENERATION_MODEL, truncate_prompt_for_generation, estimate_tokens

# Logger setup
_loggers = {}


def _setup_logger(
    logs_dir: str = "outputs/logs",
    perturbation_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> logging.Logger:
    """配置 logger 并写入文件和控制台。"""
    global _loggers
    
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        logs_path = project_root / logs_dir
    logs_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    log_filename_parts = ["generation", timestamp]
    if perturbation_type and perturbation_type != "none":
        log_filename_parts.append(perturbation_type)
    if intensity and intensity != "none":
        log_filename_parts.append(intensity)
    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = logs_path / log_filename
    
    log_key = str(log_file)
    if log_key in _loggers:
        return _loggers[log_key]
    
    logger_name = f"generation_{timestamp}_{perturbation_type or 'none'}_{intensity or 'none'}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
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
    """返回 logger 实例。"""
    return _setup_logger(perturbation_type=perturbation_type, intensity=intensity)


def build_samples_from_dataset(
    dataset_path: Path,
    limit: Optional[int] = None,
    context_window: int = 6,
    per_dialog_limit: int = 2,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从 ESConv 数据集构建样本。
    
    Args:
        dataset_path: 数据集路径（JSON 格式）
        limit: 采样数量限制
        context_window: 上下文窗口大小
        per_dialog_limit: 每个对话最多抽取的样本数
        random_seed: 随机种子
        
    Returns:
        样本列表
    """
    ensure_nltk()
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    samples: List[Dict] = []
    use_all = (limit is None) or (limit <= 0)
    per_dialog_all = (per_dialog_limit is None) or (per_dialog_limit <= 0)
    
    if random_seed is not None:
        random.seed(random_seed)
    
    for conv_id, conv in enumerate(dataset):
        used = 0
        dialog = conv.get("dialog", [])
        
        for idx, turn in enumerate(dialog):
            if turn.get("speaker") != "supporter":
                continue
            if idx == 0:
                continue
            
            # 获取上下文
            context_slice = dialog[max(0, idx - context_window) : idx]
            if not context_slice or context_slice[-1].get("speaker") != "seeker":
                continue
            
            # 格式化上下文
            context_turns = [format_turn(t) for t in context_slice]
            prompt = build_generation_prompt(context_turns)
            reference = turn.get("content", "").strip()
            
            samples.append({
                "conversation_id": conv_id,
                "supporter_turn_index": idx,
                "emotion_type": conv.get("emotion_type"),
                "problem_type": conv.get("problem_type"),
                "context_turns": context_turns,
                "reference": reference,
                "prompt": prompt,
            })
            
            used += 1
            if (not use_all) and len(samples) >= limit:
                return samples
            if (not per_dialog_all) and used >= per_dialog_limit:
                break
    
    return samples


def run_generation(
    samples: List[Dict[str, Any]],
    dataset_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 3,
    perturbation_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    运行生成推理。
    
    Args:
        samples: 样本列表
        dataset_id: 数据集ID
        temperature: 模型温度参数
        max_tokens: 最大生成token数
        max_retries: 最大重试次数
        perturbation_type: 扰动类型
        intensity: 扰动强度
        
    Returns:
        生成结果列表
    """
    logger = get_logger(perturbation_type=perturbation_type, intensity=intensity)
    logger.info(
        f"Starting generation - dataset: {dataset_id}, "
        f"samples: {len(samples)}, model: {GENERATION_MODEL}, temperature: {temperature}"
    )
    if perturbation_type or intensity:
        logger.info(f"Perturbation: type={perturbation_type or 'none'}, intensity={intensity or 'none'}")
    
    results: List[Dict[str, Any]] = []
    iterator = tqdm(
        samples,
        desc=f"Generation[{dataset_id}]",
        unit="sample",
        leave=False,
        total=len(samples),
    )
    
    error_count = 0
    for idx, sample in enumerate(iterator):
        try:
            prompt = sample.get("prompt", "")
            reference = sample.get("reference", "")
            
            logger.debug(f"Processing sample {idx+1}/{len(samples)}")
            
            # 检查并截断prompt（如果超出token限制）
            from .settings import MAX_PROMPT_TOKENS
            original_prompt_tokens = estimate_tokens(prompt)
            if original_prompt_tokens > MAX_PROMPT_TOKENS:  # 如果超过限制，进行截断
                logger.warning(
                    f"Sample {idx+1} prompt too long ({original_prompt_tokens} tokens), "
                    f"truncating to fit context limit ({MAX_PROMPT_TOKENS} tokens)..."
                )
                prompt = truncate_prompt_for_generation(prompt, max_tokens=MAX_PROMPT_TOKENS)
                truncated_tokens = estimate_tokens(prompt)
                logger.info(f"Truncated to {truncated_tokens} tokens")
            
            # 调用模型生成（带重试机制）
            response = None
            for attempt in range(max_retries):
                try:
                    completion = CLIENT.chat.completions.create(
                        model=GENERATION_MODEL,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    response = completion.choices[0].message.content.strip()
                    if response:
                        break
                    else:
                        logger.warning(f"Sample {idx+1} returned empty content (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            time.sleep(wait_time)
                except Exception as err:
                    error_msg = str(err)
                    # 检查是否是token超限错误
                    if "context_length_exceeded" in error_msg or "maximum context length" in error_msg.lower():
                        logger.warning(f"Token limit exceeded for sample {idx+1}, attempting more aggressive truncation...")
                        # 更激进的截断（保留更少的内容，使用更小的限制）
                        aggressive_max_tokens = int(MAX_PROMPT_TOKENS * 0.5)  # 使用50%的限制作为更激进的截断
                        prompt = truncate_prompt_for_generation(prompt, max_tokens=aggressive_max_tokens)
                        if attempt < max_retries - 1:
                            continue  # 重试
                    logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {err}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                    else:
                        # 最后一次尝试失败，抛出异常
                        raise
            
            if not response:
                logger.warning(f"Sample {idx+1} returned empty content")
            
            results.append({
                "conversation_id": sample.get("conversation_id"),
                "supporter_turn_index": sample.get("supporter_turn_index"),
                "emotion_type": sample.get("emotion_type"),
                "problem_type": sample.get("problem_type"),
                "context_turns": sample.get("context_turns"),
                "reference": reference,
                "generated": response,
                "prompt": prompt,
            })
        except Exception as e:
            error_count += 1
            logger.error(f"Error while processing sample {idx+1}: {str(e)}", exc_info=True)
            results.append({
                "conversation_id": sample.get("conversation_id"),
                "supporter_turn_index": sample.get("supporter_turn_index"),
                "reference": sample.get("reference", ""),
                "generated": None,
                "error": str(e),
            })
    
    logger.info(
        f"Generation completed - total: {len(samples)}, "
        f"succeeded: {len(samples) - error_count}, failed: {error_count}"
    )
    return results


def score_generation(
    generation_results: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    filename_base: Optional[str] = None,
    perturbation_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    计算生成任务的评估指标。
    
    Args:
        generation_results: 生成结果列表
        output_dir: 输出目录
        filename_base: 文件名基础
        perturbation_type: 扰动类型
        intensity: 扰动强度
        
    Returns:
        包含所有指标的字典
    """
    logger = get_logger(perturbation_type=perturbation_type, intensity=intensity)
    logger.info(f"Start scoring - samples: {len(generation_results)}")
    
    ensure_nltk()
    
    # 准备参考和生成文本
    references = []
    hypotheses = []
    conversation_details = []
    
    for idx, result in enumerate(generation_results):
        reference = result.get("reference", "")
        generated = result.get("generated", "")
        
        if generated is None:
            logger.warning(f"Sample {idx+1} has no generated text, skipping")
            continue
        
        references.append([reference])  # 每个样本一个参考
        hypotheses.append(generated)
        
        # 计算单条指标
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(generated.lower())
        
        conversation_details.append({
            "index": idx,
            "conversation_id": result.get("conversation_id"),
            "supporter_turn_index": result.get("supporter_turn_index"),
            "emotion_type": result.get("emotion_type"),
            "problem_type": result.get("problem_type"),
            "reference": reference,
            "generated": generated,
            "reference_len": len(ref_tokens),
            "generated_len": len(hyp_tokens),
        })
    
    if not references or not hypotheses:
        logger.warning("No valid samples for scoring")
        return {
            "aggregate": {},
            "per_sample": {},
            "conversation_details": conversation_details,
        }
    
    # 使用 Metric 类计算指标
    metric = Metric(toker=None)
    for refs, hyp in zip(references, hypotheses):
        metric.forward(refs, hyp, chinese=False)
    
    aggregate_metrics, per_sample_metrics = metric.close()
    
    # 添加单条 F1 和 ROUGE-L 到 conversation_details
    f1_scores = per_sample_metrics.get("f1", [])
    rouge_l_scores = per_sample_metrics.get("rouge-l", [])
    
    for detail, f1_score, rl_score in zip(
        conversation_details,
        f1_scores,
        rouge_l_scores,
    ):
        detail["sample_f1"] = float(f1_score)
        detail["sample_rouge_l"] = float(rl_score)
    
    # 进行 LLM-as-Judge 评估
    logger.info("=" * 60)
    logger.info("Starting LLM-as-Judge evaluation...")
    logger.info("=" * 60)
    
    judge_metrics = {}
    judge_per_sample = {}
    
    try:
        from .judge import (
            convert_esconv_to_evaluation_format,
            evaluate_single_sample,
        )
        from .settings import JUDGE_METRICS
        from .settings import JUDGE_MODEL
        import time
        
        # 转换格式并评估
        judge_results = []
        iterator = tqdm(
            generation_results,
            desc="LLM-as-Judge evaluation",
            unit="sample",
            leave=False,
            total=len(generation_results),
        )
        
        for idx, gen_result in enumerate(iterator):
            if gen_result.get("generated") is None:
                continue
            
            try:
                eval_sample = convert_esconv_to_evaluation_format(gen_result)
                judge_result = evaluate_single_sample(eval_sample, judge_model=JUDGE_MODEL, verbose=False)
                judge_results.append(judge_result)
                
                # 添加到 conversation_details
                for detail in conversation_details:
                    if (detail.get("conversation_id") == judge_result.get("conversation_id") and
                        detail.get("supporter_turn_index") == judge_result.get("supporter_turn_index")):
                        detail["judge_scores"] = judge_result.get("evaluation_scores", {})
                        break
            except Exception as e:
                logger.warning(f"评估样本 {idx+1} 失败: {e}", exc_info=True)
                # 即使评估失败，也继续处理下一个样本
                continue
        
        # 计算 judge 指标的聚合统计
        for metric_name in JUDGE_METRICS:
            scores = [
                r.get("evaluation_scores", {}).get(metric_name)
                for r in judge_results
                if r.get("evaluation_scores", {}).get(metric_name) is not None
            ]
            
            if scores:
                judge_metrics[f"judge_{metric_name.lower()}"] = {
                    "mean": float(sum(scores) / len(scores)),
                    "min": float(min(scores)),
                    "max": float(max(scores)),
                    "std": float((sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5),
                }
                # 同时添加到 aggregate_metrics 中（用于汇总报告）
                aggregate_metrics[f"judge_{metric_name.lower()}"] = judge_metrics[f"judge_{metric_name.lower()}"]["mean"]
        
        # 计算 per_sample judge 指标
        for metric_name in JUDGE_METRICS:
            per_sample_scores = [
                r.get("evaluation_scores", {}).get(metric_name)
                for r in judge_results
            ]
            judge_per_sample[f"judge_{metric_name.lower()}"] = per_sample_scores
        
        logger.info("=" * 60)
        logger.info("LLM-as-Judge evaluation completed")
        logger.info("=" * 60)
        for metric_name, stats in judge_metrics.items():
            logger.info(f"  {metric_name}: {stats['mean']:.4f} (std: {stats['std']:.4f}, range: {stats['min']:.4f}-{stats['max']:.4f})")
        
    except Exception as e:
        logger.warning(f"LLM-as-Judge evaluation failed: {e}", exc_info=True)
        logger.warning("Continuing without judge metrics...")
    
    result = {
        "aggregate": aggregate_metrics,
        "per_sample": per_sample_metrics,
        "conversation_details": conversation_details,
        "judge_metrics": judge_metrics,
        "judge_per_sample": judge_per_sample,
    }
    
    logger.info("=" * 60)
    logger.info("Evaluation summary:")
    for key, value in aggregate_metrics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    logger.info("=" * 60)
    
    return result


def format_generation_summary(metrics: Dict[str, Any]) -> str:
    """格式化生成指标为可读的文本摘要。"""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("ESConv Generation Report")
    lines.append("=" * 80)
    lines.append("")
    
    aggregate = metrics.get("aggregate", {})
    if aggregate:
        lines.append("[Overall Metrics]")
        for key, value in aggregate.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
    
    conversation_details = metrics.get("conversation_details", [])
    if conversation_details:
        lines.append(f"[Sample Details]")
        lines.append(f"  Total samples: {len(conversation_details)}")
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)
