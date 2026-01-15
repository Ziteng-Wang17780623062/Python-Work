"""
论理性测试的核心流程：
1) 加载并采样数据（保证每次采样不同）。
2) 构造提示词，调用大模型，使用 tqdm 显示进度。
3) 保存结果，命名规则参考 C-SSRS。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .data_utils import load_dataset, save_results
from .llm import call_model
import json

from .prompts import build_reasoning_prompt, build_judge_prompt
from .settings import (
    DATASET_PATH,
    EVALUATION_MODEL,
    OUTPUT_DIR,
    DEFAULT_TEMPERATURE,
    JUDGE_MODEL,
)


def run_reasoning_evaluation(
    dataset_path: Path = DATASET_PATH,
    sample_size: int = 10,
    model: str = EVALUATION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    执行完整的论理性评估流程。
    返回的结果字典将用于保存到文件。
    """
    samples = load_dataset(dataset_path, sample_size=sample_size, random_seed=random_seed)

    results: List[Dict[str, Any]] = []
    iterator = tqdm(
        samples,
        desc="运行大模型推理",
        unit="sample",
        total=len(samples),
    )

    for sample in iterator:
        prompt = build_reasoning_prompt(sample)
        response = call_model(prompt=prompt, model=model, temperature=temperature)

        results.append(
            {
                "sample_id": sample.get("id"),
                "subcategory": sample.get("subcategory"),
                "scenario": sample.get("scenario"),
                "options": sample.get("options"),
                "reasoning_task": sample.get("reasoning_task"),
                "expected_reasoning": sample.get("expected_reasoning"),
                "model_behavior": sample.get("model_behavior"),
                "prompt": prompt,
                "response": response,
            }
        )

    # LLM-as-judge 评分
    judged = []
    judge_iterator = tqdm(
        results,
        desc="Judging responses",
        unit="sample",
        total=len(results),
    )
    for item in judge_iterator:
        model_reply = item.get("response", {}).get("content") or ""
        judge_prompt = build_judge_prompt(
            sample={
                "scenario": item.get("scenario", ""),
                "options": item.get("options", ""),
                "reasoning_task": item.get("reasoning_task", ""),
                "expected_reasoning": item.get("expected_reasoning", ""),
                "model_behavior": item.get("model_behavior", ""),
            },
            model_reply=model_reply,
        )
        judge_resp = call_model(prompt=judge_prompt, model=JUDGE_MODEL, temperature=0.0)
        judge_parsed = _parse_json_content(judge_resp.get("content"))

        judged.append(
            {
                "sample_id": item.get("sample_id"),
                "subcategory": item.get("subcategory"),
                "assistant_response": item.get("response"),
                "judge_prompt": judge_prompt,
                "judge_raw": judge_resp,
                "judge_parsed": judge_parsed,
            }
        )

    # 计算 6 个维度的平均分（sc1-sc5, sim），作为模型综合表现
    avg_scores = _compute_average_scores(judged)

    metadata = {
        "task": "ethics_reasoning",
        "dataset": str(dataset_path),
        "dataset_id": Path(dataset_path).name,
        "samples": len(samples),
        "model": model,
        "temperature": temperature,
        "timestamp": datetime.utcnow().isoformat(),
        "judge_model": JUDGE_MODEL,
    }

    # 将平均分写入 metadata，方便下游脚本直接使用
    if avg_scores:
        metadata["avg_scores"] = avg_scores

    output = {
        **metadata,
        "results": results,
        "judged": judged,
    }

    summary_text = _build_summary_text(metadata, judged)

    saved_path = save_results(output, output_dir=OUTPUT_DIR, summary_text=summary_text)
    output["saved_path"] = str(saved_path)

    # 终端打印最终统计结果
    print("\n=== Summary ===")
    print(summary_text)
    print(f"Saved: {saved_path}")
    return output


def _compute_average_scores(judged: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    基于裁判模型解析结果，计算 6 个评分维度（sc1-sc5, sim）的平均分。

    - 仅对能够成功解析为浮点数的分数参与平均；
    - 若某个维度在所有样本中均缺失或无法解析，则该维度不出现在返回结果中。
    """
    if not judged:
        return {}

    score_keys = ["sc1", "sc2", "sc3", "sc4", "sc5", "sim"]
    sums = {k: 0.0 for k in score_keys}
    counts = {k: 0 for k in score_keys}

    for item in judged:
        parsed = item.get("judge_parsed") or {}
        for key in score_keys:
            if key not in parsed:
                continue
            value = parsed.get(key)
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            sums[key] += v
            counts[key] += 1

    averages: Dict[str, float] = {}
    for key in score_keys:
        if counts[key] > 0:
            averages[key] = sums[key] / counts[key]

    return averages


def _parse_json_content(content: str) -> Dict[str, Any]:
    """
    尝试从裁判模型输出中解析 JSON。
    - 支持 Markdown 代码块
    - 支持纯 JSON
    - 回退：如果只包含一个数字，则视为 overall 分数
    解析失败则返回 {"unparsed": 原文}
    """
    if not content:
        return {}

    text = content.strip()

    # 1) 代码块中的 JSON（兼容 ```json 开头）
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 处理以 "json" 开头的情况
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                try:
                    return json.loads(part)
                except Exception:
                    pass

    # 2) 直接尝试解析整体 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3) 回退：如果内容就是一个数字或只包含一个数字，视为 overall
    stripped = text.strip()
    try:
        value = float(stripped)
        return {"overall": value}
    except ValueError:
        # 尝试从文本中提取第一个数字
        import re

        match = re.search(r"(-?\d+(?:\.\d+)?)", stripped)
        if match:
            try:
                return {"overall": float(match.group(1))}
            except ValueError:
                pass

    # 4) 全部失败，记录原文
    return {"unparsed": content}


def _build_summary_text(metadata: Dict[str, Any], judged: List[Dict[str, Any]]) -> str:
    """
    生成同名文本摘要。

    - 在终端和文本摘要中给出 6 个维度（sc1-sc5, sim）的平均分；
    - 同时逐条列出每个样本的 6 维分数，便于人工快速查看。
    """

    lines = [
        f"Task: {metadata.get('task')}",
        f"Dataset: {metadata.get('dataset_id')}",
        f"Samples: {metadata.get('samples')}",
        f"Model: {metadata.get('model')}",
        f"Judge model: {metadata.get('judge_model')}",
        f"Temperature: {metadata.get('temperature')}",
        f"Timestamp: {metadata.get('timestamp')}",
        "",
    ]
    lines.append(f"Judged count: {len(judged)}")

    # 1) 打印 6 个维度的平均分
    avg_scores = metadata.get("avg_scores") or {}
    if avg_scores:
        lines.append("")
        lines.append("Average scores over all judged samples:")
        for key in ["sc1", "sc2", "sc3", "sc4", "sc5", "sim"]:
            if key in avg_scores:
                lines.append(f"- {key}: {avg_scores[key]:.4f}")
    else:
        lines.append("")
        lines.append("Average scores over all judged samples: <not available>")

    # 逐条打印所有分数，便于在终端直接查看最终 6 维成绩
    if judged:
        lines.append("")
        lines.append("Per-sample scores (from judge_parsed):")
        for idx, item in enumerate(judged, start=1):
            sid = item.get("sample_id", f"sample_{idx}")
            parsed = item.get("judge_parsed") or {}
            lines.append(f"- sample_id: {sid}")
            if not parsed:
                lines.append("  scores: <no parsable scores>")
                continue
            # 只展示 6 个评分维度的数值分数（sc1-sc5, sim），
            # 不在终端打印 *_reason 之类的原因字段，原因仅保存在 JSON 中。
            for key in ["sc1", "sc2", "sc3", "sc4", "sc5", "sim"]:
                if key in parsed:
                    lines.append(f"  {key}: {parsed[key]}")
    return "\n".join(lines)


__all__ = ["run_reasoning_evaluation"]
