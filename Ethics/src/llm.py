"""
统一封装大模型调用，便于后续替换提供方。
参考 C-SSRS 中的调用方式，默认使用 chat.completions.create。
"""

from typing import Any, Dict

from openai import OpenAIError

from .settings import CLIENT


def call_model(
    prompt: str,
    model: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    调用大模型，返回统一结构，方便后续保存。
    """
    try:
        completion = CLIENT.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        message = completion.choices[0].message
        return {
            "content": message.content,
            "raw": completion.model_dump(),
        }
    except OpenAIError as e:
        return {
            "content": None,
            "error": str(e),
        }


__all__ = ["call_model"]
