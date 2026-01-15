"""
ESConv 数据集生成提示词构建。
"""

from typing import Dict, List


def format_turn(turn: Dict[str, str]) -> str:
    """格式化对话轮次。
    
    Args:
        turn: 包含 speaker 和 content 的字典
        
    Returns:
        格式化后的字符串，如 "Seeker: ..." 或 "Supporter: ..."
    """
    prefix = "Seeker" if turn.get("speaker") == "seeker" else "Supporter"
    return f"{prefix}: {turn.get('content', '').strip()}"


def build_generation_prompt(context_turns: List[str]) -> str:
    """
    构建 ESConv 数据集的生成提示词。
    
    Args:
        context_turns: 上下文对话轮次列表（已格式化的字符串列表）
        
    Returns:
        格式化的提示词字符串
    """
    # 将上下文对话合并
    context = "\n".join(context_turns) if context_turns else ""
    
    # 添加 "Supporter:" 提示，让模型生成支持者的回复
    prompt = f"""{context}
Supporter:"""
    
    return prompt
