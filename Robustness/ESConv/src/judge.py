import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from .settings import (
    JUDGE_CLIENT,
    JUDGE_MODEL,
    JUDGE_METRICS,
    JUDGE_PROMPT_TEMPLATES,
    truncate_judge_prompt_components,
    estimate_tokens,
)


def extract_score(response: str) -> Optional[float]:
    """从模型响应中提取0-5的分数"""
    # 尝试匹配数字（包括小数）
    match = re.search(r'\b([0-5](?:\.\d+)?)\b', response.strip())
    if match:
        score = float(match.group(1))
        # 确保分数在0-5范围内
        if 0 <= score <= 5:
            return score
    return None


def call_gpt4_for_evaluation(prompt: str, max_retries: int = 3) -> Optional[float]:
    """调用 judge 模型进行单次评估（带重试机制）"""
    # 检查prompt长度，如果超出限制则截断
    original_tokens = estimate_tokens(prompt)
    if original_tokens > 120000:  # 如果超过限制，进行截断
        print(f"警告: Prompt过长 ({original_tokens} tokens), 进行截断...")
        # 对于评估prompt，我们需要保留关键部分
        # 简单截断：保留末尾部分（通常包含最重要的评估内容）
        from .settings import truncate_text, MAX_PROMPT_TOKENS
        prompt = truncate_text(prompt, MAX_PROMPT_TOKENS, keep_end=True)
        truncated_tokens = estimate_tokens(prompt)
        print(f"已截断至 {truncated_tokens} tokens")
    
    for attempt in range(max_retries):
        try:
            response = JUDGE_CLIENT.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            content = response.choices[0].message.content.strip()
            score = extract_score(content)
            if score is not None:
                return score
            else:
                # 无法提取有效分数，记录警告并重试
                print(f"警告: 无法从响应中提取有效分数: {content}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        except Exception as e:
            error_msg = str(e)
            # 检查是否是token超限错误
            if "context_length_exceeded" in error_msg or "maximum context length" in error_msg.lower():
                print(f"Token限制超出，进行更激进的截断...")
                from .settings import truncate_text, MAX_PROMPT_TOKENS
                prompt = truncate_text(prompt, 100000, keep_end=True)  # 更激进的截断
                if attempt < max_retries - 1:
                    continue  # 重试
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
    return None


def extract_context_and_user_input(conversation_text: str) -> Tuple[str, str]:
    """从conversation_text中提取Context和User_input
    
    conversation_text格式示例:
    "Seeker: Hello\nSupporter: Hello, what would you like to talk about?\nSeeker: I am having anxiety\nSupporter: I understand..."
    
    返回:
    - context: 最后一个Seeker输入之前的所有对话（不包括最后一个Supporter回复）
    - user_input: 最后一个Seeker的输入内容
    """
    lines = conversation_text.split('\n')
    
    # 找到最后一个Seeker的输入
    last_seeker_line = None
    last_seeker_idx = -1
    
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.startswith('Seeker:'):
            last_seeker_line = line
            last_seeker_idx = i
            break
    
    if last_seeker_line is None:
        # 如果没有找到Seeker输入，返回空
        return conversation_text, ""
    
    # 提取User_input（最后一个Seeker输入的内容）
    user_input = last_seeker_line.replace('Seeker:', '').strip()
    
    # Context是最后一个Seeker输入之前的所有对话
    # 注意：conversation_text包含了model_response（最后一个Supporter回复），
    # 但Context应该只包含到最后一个Seeker输入为止
    context_lines = lines[:last_seeker_idx]
    context = '\n'.join(context_lines).strip()
    
    return context, user_input


def convert_esconv_to_evaluation_format(generation_result: Dict) -> Dict:
    """将 ESConv 生成结果转换为评估格式"""
    context_turns = generation_result.get("context_turns", [])
    conversation_text = "\n".join(context_turns) if isinstance(context_turns, list) else str(context_turns)
    
    return {
        "conversation_id": generation_result.get("conversation_id"),
        "supporter_turn_index": generation_result.get("supporter_turn_index"),
        "emotion_type": generation_result.get("emotion_type", ""),
        "problem_type": generation_result.get("problem_type", ""),
        "conversation_text": conversation_text,
        "reference_response": generation_result.get("reference", ""),
        "model_response": generation_result.get("generated", ""),
    }


def evaluate_single_sample(sample: Dict, judge_model: Optional[str] = None, verbose: bool = False) -> Dict:
    """评估单个样本"""
    if judge_model is None:
        judge_model = JUDGE_MODEL
    
    context, user_input = extract_context_and_user_input(sample.get("conversation_text", ""))
    reference_response = sample.get("reference_response", "")
    model_response = sample.get("model_response", "")
    
    scores = {}
    
    if verbose:
        print(f"评估样本 {sample.get('conversation_id', '?')}-{sample.get('supporter_turn_index', '?')}...")
    
    for metric in JUDGE_METRICS:
        try:
            # 在格式化prompt之前，先检查并截断各个组件
            prompt_template = JUDGE_PROMPT_TEMPLATES[metric]
            
            # 估算完整prompt的token数
            test_prompt = prompt_template.format(
                Context=context,
                User_input=user_input,
                GT_Response=reference_response,
                Pred_Response=model_response
            )
            estimated_tokens = estimate_tokens(test_prompt)
            
            # 如果超出限制，截断各个组件
            if estimated_tokens > 120000:
                if verbose:
                    print(f"  评估 {metric}... (原始token数: {estimated_tokens}, 进行截断...)")
                context, user_input, reference_response, model_response = truncate_judge_prompt_components(
                    context, user_input, reference_response, model_response, prompt_template
                )
                # 重新估算
                test_prompt = prompt_template.format(
                    Context=context,
                    User_input=user_input,
                    GT_Response=reference_response,
                    Pred_Response=model_response
                )
                estimated_tokens = estimate_tokens(test_prompt)
                if verbose:
                    print(f"    截断后token数: {estimated_tokens}")
            else:
                if verbose:
                    print(f"  评估 {metric}...", end=" ")
            
            prompt = prompt_template.format(
                Context=context,
                User_input=user_input,
                GT_Response=reference_response,
                Pred_Response=model_response
            )
            
            score = call_gpt4_for_evaluation(prompt, max_retries=3)
            scores[metric] = score
            
            if verbose:
                if score is not None:
                    print(f"得分: {score}")
                else:
                    print("失败")
        except Exception as e:
            # 如果评估某个指标失败，记录错误但继续评估其他指标
            if verbose:
                print(f"失败: {e}")
            scores[metric] = None
        
        time.sleep(0.5)  # 避免API限流
    
    return {**sample, "evaluation_scores": scores}


def main():
    """命令行工具：独立运行 judge 评估"""
    parser = argparse.ArgumentParser(description="使用 judge 模型评估ESConv生成结果")
    parser.add_argument("--input", type=Path, default=None, help="输入的ESConv结果JSON文件路径")
    parser.add_argument("--output", type=Path, default=None, help="输出结果文件路径")
    parser.add_argument("--judge_model", type=str, default=None, help=f"评估模型（默认: {JUDGE_MODEL}）")
    parser.add_argument("--limit", type=int, default=None, help="仅评估前N条样本")
    args = parser.parse_args()
    
    if args.judge_model is None:
        args.judge_model = JUDGE_MODEL
    
    if args.input is None:
        results_dir = Path(__file__).parent.parent / "outputs" / "results"
        if results_dir.exists():
            json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            json_files = [f for f in json_files if "summary" not in f.name and "robustness" not in f.name]
            if json_files:
                args.input = json_files[0]
                print(f"自动选择: {args.input}")
            else:
                print("错误: 未找到结果文件")
                return
        else:
            print("错误: 未找到结果目录")
            return
    
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_judge_evaluation.json"
    
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    generation_results = data.get("generation", data if isinstance(data, list) else [])
    if args.limit:
        generation_results = generation_results[:args.limit]
    
    print(f"共 {len(generation_results)} 条样本需要评估\n")
    
    evaluation_samples = [convert_esconv_to_evaluation_format(r) for r in generation_results]
    results = []
    for idx, sample in enumerate(evaluation_samples, 1):
        print(f"[{idx}/{len(evaluation_samples)}] ", end="")
        result = evaluate_single_sample(sample, judge_model=args.judge_model, verbose=True)
        results.append(result)
        print()
    
    avg_scores = {}
    for metric in JUDGE_METRICS:
        scores = [r["evaluation_scores"][metric] for r in results if r["evaluation_scores"].get(metric) is not None]
        if scores:
            avg_scores[metric] = round(sum(scores) / len(scores), 2)
    
    output_data = {
        "judge_model": args.judge_model,
        "total_samples": len(results),
        "statistics": {"average_scores": avg_scores},
        "evaluation_results": [
            {
                "conversation_id": r.get("conversation_id"),
                "supporter_turn_index": r.get("supporter_turn_index"),
                "evaluation_scores": r.get("evaluation_scores", {})
            }
            for r in results
        ]
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估完成！结果保存至: {args.output}")
    print(f"\n平均分数:")
    for metric in JUDGE_METRICS:
        if avg_scores.get(metric) is not None:
            print(f"  {metric}: {avg_scores[metric]:.2f}")


if __name__ == "__main__":
    main()

