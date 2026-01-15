"""
被测试LLM生成回复模块。
"""

from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .settings import TEST_MODEL, CLIENT


def generate_response(prompt: str, model: Optional[str] = None) -> str:
    """
    使用被测试LLM自由生成回复。
    
    测试逻辑：
    1. 使用数据集中的prompt，让被测试的LLM自由生成回复
    2. 然后使用judge LLM进行评估，输出0/1
    
    Args:
        prompt: 用户问题（来自数据集中的prompt列）
        model: 模型名称（如果为None，使用settings中的TEST_MODEL）
    
    Returns:
        被测试LLM自由生成的回复（无任何限制或引导）
    """
    # TODO: 实现LLM调用逻辑
    # 这里先返回占位符
    if model is None:
        model = TEST_MODEL
    
    if model is None:
        raise ValueError("未配置被测试的LLM模型")
    
    # 调用LLM API
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def generate_responses_for_samples(
    samples: List[Dict[str, Any]],
    model: Optional[str] = None
) -> List[str]:
    """
    为所有样本使用被测试LLM自由生成回复。
    
    测试逻辑：
    1. 使用数据集中的prompt，让被测试的LLM自由生成回复
    2. 然后使用judge LLM进行评估，输出0/1
    
    Args:
        samples: 样本列表（包含数据集中的prompt）
        model: 模型名称
    
    Returns:
        被测试LLM自由生成的回复列表（与样本顺序对应）
    """
    responses = []
    for sample in tqdm(samples, desc="生成LLM回复", unit="样本"):
        prompt = sample.get("prompt", "")
        response = generate_response(prompt, model)
        responses.append(response)
    return responses

