"""
ESConv 项目全局配置。
"""

from pathlib import Path
from typing import Tuple
from openai import OpenAI

# 项目路径配置
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# API/模型配置
import os
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 模型配置
GENERATION_MODEL = "deepseek-v3.1-250821"

# LLM-as-Judge 评估配置（与 evaluate_with_gpt4.py 保持一致）
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY", "")
JUDGE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_CLIENT = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

# LLM-as-Judge 评估指标
JUDGE_METRICS = ["Fluency", "Professionalism", "Empathy", "Helpfulness"]

# LLM-as-Judge 评估输入部分模板（用于多个指标）
_JUDGE_INPUT_SECTION = """## Input
### Context: {Context}

### Seeker's Input: {User_input}

### Model's Response
Ground Truth Response(reference): {GT_Response}
Model's Response: {Pred_Response}"""

# LLM-as-Judge 评估提示词模板（与 evaluate_with_gpt4.py 完全一致）
JUDGE_PROMPT_TEMPLATES = {
    "Fluency": """# System
You are an impartial judge, familiar with psychology knowledge and psychological counseling.

## Attention
You need to evaluate the **Fluency** of the AI assistant's responses based on its dialogue with the user.
Your evaluation results must be based on the Evaluation Criteria.

## Evaluation Criteria
1.**Fluency [0-5 points]**: Fluency evaluates whether language expression is natural, coherent, and comprehensible.
    - 0: Incoherent or difficult to understand; contains grammar or logic issues.
    - 1: Unclear expression; user may struggle to grasp the meaning.
    - 2: Some parts are confusing, though the main point can be inferred.
    - 3: Mostly clear and coherent with minor ambiguities.
    - 4: Fluent and well-structured; logically organized and easy to follow.
    - 5: Concise and impactful language; precise and elegant communication that conveys ideas efficiently.

## Constraints
- Avoid evaluation bias due to preference for specific model names.
- Avoid evaluation bias due to response length.

## Input
### Context: {Context}

### Seeker's Input: {User_input}

### Model's Response
Ground Truth Response(reference): {GT_Response}
Model's Response: {Pred_Response}

Based on the rules, give your Fluency score (The number only) to the Model's Response.

## Output
Fluency score (The number only)""",

    "Professionalism": """# System
You are an impartial judge, familiar with psychology knowledge and psychological counseling.

## Attention
You need to evaluate the **Professionalism** of the AI assistant's responses based on its dialogue with the user.
Your evaluation results must be based on the Evaluation Criteria.

## Evaluation Criteria
1.**Professionalism [0-5 points]**: Evaluates whether the model demonstrates psychological knowledge, follows ethical principles, and avoids misleading or inappropriate advice.
    - 0: Contains harmful, misleading, or clearly inappropriate content that may violate ethical or psychological guidelines.
    - 1: Shows serious misunderstanding or misuse of psychological concepts, or provides inappropriate advice.
    - 2: Minor factual inaccuracies or advice that lacks evidence, but does not pose direct harm.
    - 3: No major errors; advice is acceptable and somewhat aligned with psychological principles.
    - 4: Demonstrates solid understanding of psychological concepts and appropriate intervention techniques.
    - 5: Highly professional, reflects strong psychological insight, maintains boundaries, and communicates in a grounded, ethical manner.

## Constraints
- Avoid evaluation bias due to preference for specific model names.
- Avoid evaluation bias due to response length.
- Do not confuse professionalism with warmth or empathy—focus on psychological accuracy and appropriateness.

## Input
### Context: {Context}

### Seeker's Input: {User_input}

### Model's Response
Ground Truth Response(reference): {GT_Response}
Model's Response: {Pred_Response}

Based on the rules, give your Professionalism score (The number only) to the Model's Response.

## Output
Professionalism score (The number only)""",

    "Empathy": """# System
You are an impartial judge, familiar with psychology knowledge and psychological counseling.

## Attention
You need to evaluate the **Empathy** of the AI assistant's responses based on its dialogue with the user.
Your evaluation results must be based on the Evaluation Criteria.

## Evaluation Criteria
1.**Empathy [0-5 points]**: Evaluates whether the model genuinely understands the user's emotions, expresses care, and provides emotional support.
    - 0: Contains statements that may harm the user emotionally or lead to a negative emotional trajectory.
    - 1: Fails to provide emotional comfort or assist the user in analyzing their problems.
    - 2: Either lacks emotional comfort or fails to support problem analysis.
    - 3: No significant issues, but empathy and analysis remain surface-level.
    - 4: Demonstrates a warm, human-like tone—like a friend—offering both emotional relief and analytical support.
    - 5: Deep emotional insight with sincere and stable empathy, conveyed through attentive and flexible language.

## Constraints
- Avoid evaluation bias due to preference for specific model names.
- Avoid evaluation bias due to response length.

## Input
### Context: {Context}

### Seeker's Input: {User_input}

### Model's Response
Ground Truth Response(reference): {GT_Response}
Model's Response: {Pred_Response}

Based on the rules, give your Empathy score (The number only) to the Model's Response.

## Output
Empathy score (The number only)""",

    "Helpfulness": """# System
You are an impartial judge, familiar with psychology knowledge and psychological counseling.

## Attention
You need to evaluate the **Helpfulness** of the AI assistant's responses based on its dialogue with the user.
Your evaluation results must be based on the Evaluation Criteria.

## Evaluation Criteria
1.**Helpfulness [0-5 points]**: Helpfulness evaluates the effectiveness of an AI assistant's suggestions by considering both the number of recommendations provided per interaction and the relevance or usefulness of each suggestion in addressing the user's question.
    - 0: Irrelevant, misleading, or potentially harmful suggestions.
    - 1: Ineffective or generic advice that does not respond to the user's needs.
    - 2: Weakly relevant suggestions with limited practical value.
    - 3: Somewhat helpful; suggestions are relevant and usable.
    - 4: Clear and practical advice that aligns well with the user's issue.
    - 5: Highly insightful, tailored, and actionable suggestions that offer strong guidance and value.

## Constraints
- Avoid evaluation bias due to preference for specific model names.
- Avoid evaluation bias due to response length.

## Input
### Context: {Context}

### Seeker's Input: {User_input}

### Model's Response
Ground Truth Response(reference): {GT_Response}
Model's Response: {Pred_Response}

Based on the rules, give your Helpfulness score (The number only) to the Model's Response.

## Output
Helpfulness score (The number only)"""
}

# ESConv 数据集问题类型定义
ESCONV_PROBLEM_TYPES = {
    "ongoing depression": "持续抑郁",
    "breakup with partner": "与伴侣分手",
    "job crisis": "工作危机",
    "problems with friends": "朋友问题",
    "academic pressure": "学业压力",
    "procrastination": "拖延症",
    "alcohol abuse": "酗酒",
    "issues with parent": "父母问题",
    "sleep problems": "睡眠问题",
    "appearance anxiety": "外貌焦虑",
    "school bullying": "校园霸凌",
    "issues with children": "子女问题",
}

# Token 限制配置
# deepseek-v3模型通常支持128K上下文，但为了安全起见，设置为64000以避免API调用失败
MAX_CONTEXT_TOKENS = 64000  # 模型最大上下文长度（保守设置，避免API调用失败）
SAFETY_MARGIN = 2000  # 安全边距，预留一些token用于系统消息和响应（增加安全边距）
MAX_PROMPT_TOKENS = MAX_CONTEXT_TOKENS - SAFETY_MARGIN  # 实际可用的prompt token数（62000）


def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量。
    使用保守的估算方法：
    - 英文：1 token ≈ 4 字符
    - 中文：1 token ≈ 1.5 字符
    - 混合文本：使用更保守的估算（约3字符/token）
    
    Args:
        text: 输入文本
        
    Returns:
        估算的token数量
    """
    if not text:
        return 0
    
    # 统计中文字符数（CJK统一汉字范围）
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    # 其他字符（包括英文、数字、标点等）
    other_chars = len(text) - chinese_chars
    
    # 估算：中文按1.5字符/token，其他按4字符/token
    # 为了更保守，使用混合估算：整体按3字符/token
    estimated_tokens = len(text) / 3
    
    # 至少返回1（即使是空字符串，系统消息也会占用token）
    return max(1, int(estimated_tokens))


def truncate_text(text: str, max_tokens: int, keep_end: bool = True) -> str:
    """
    截断文本以符合token限制。
    
    Args:
        text: 要截断的文本
        max_tokens: 最大token数
        keep_end: 如果为True，保留文本末尾部分；如果为False，保留开头部分
        
    Returns:
        截断后的文本
    """
    if not text:
        return text
    
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    
    # 计算需要保留的字符数（保守估算）
    # 使用更保守的估算：3字符/token
    max_chars = max_tokens * 3
    
    if keep_end:
        # 保留末尾部分
        return text[-max_chars:] if len(text) > max_chars else text
    else:
        # 保留开头部分
        return text[:max_chars] if len(text) > max_chars else text


def truncate_prompt_for_generation(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """
    为生成任务截断prompt。
    保留prompt的末尾部分（因为需要生成回复）。
    
    Args:
        prompt: 原始prompt
        max_tokens: 最大token数
        
    Returns:
        截断后的prompt
    """
    return truncate_text(prompt, max_tokens, keep_end=True)


def truncate_judge_prompt_components(
    context: str,
    user_input: str,
    gt_response: str,
    pred_response: str,
    prompt_template: str,
    max_tokens: int = MAX_PROMPT_TOKENS
) -> Tuple[str, str, str, str]:
    """
    为评估任务截断各个组件。
    优先保留User_input、GT_Response和Pred_Response，截断Context。
    
    Args:
        context: 上下文
        user_input: 用户输入
        gt_response: 参考回复
        pred_response: 模型回复
        prompt_template: 提示词模板（用于估算固定部分的token）
        max_tokens: 最大token数
        
    Returns:
        截断后的(context, user_input, gt_response, pred_response)
    """
    # 估算模板固定部分的token数（不包含变量部分）
    template_without_vars = prompt_template.replace("{Context}", "").replace("{User_input}", "").replace("{GT_Response}", "").replace("{Pred_Response}", "")
    template_tokens = estimate_tokens(template_without_vars)
    
    # 估算其他必需组件的token数
    user_input_tokens = estimate_tokens(user_input)
    gt_response_tokens = estimate_tokens(gt_response)
    pred_response_tokens = estimate_tokens(pred_response)
    
    # 计算可用于Context的token数
    available_for_context = max_tokens - template_tokens - user_input_tokens - gt_response_tokens - pred_response_tokens - 100  # 额外100token安全边距
    
    # 如果可用token数太少，需要进一步截断其他组件
    if available_for_context < 100:  # 至少保留100token给context
        # 按比例截断所有组件
        total_required = template_tokens + user_input_tokens + gt_response_tokens + pred_response_tokens
        scale_factor = (max_tokens - template_tokens - 100) / total_required if total_required > 0 else 1.0
        scale_factor = min(1.0, max(0.1, scale_factor))  # 限制在0.1到1.0之间
        
        user_input = truncate_text(user_input, int(user_input_tokens * scale_factor), keep_end=True)
        gt_response = truncate_text(gt_response, int(gt_response_tokens * scale_factor), keep_end=True)
        pred_response = truncate_text(pred_response, int(pred_response_tokens * scale_factor), keep_end=True)
        available_for_context = max_tokens - template_tokens - estimate_tokens(user_input) - estimate_tokens(gt_response) - estimate_tokens(pred_response) - 100
    
    # 截断Context（保留末尾部分，因为最近的对话更重要）
    context = truncate_text(context, max(100, available_for_context), keep_end=True)
    
    return context, user_input, gt_response, pred_response