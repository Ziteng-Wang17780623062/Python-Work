from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Tuple

evaluate = None
try:  # pragma: no cover - 运行时按需加载
    evaluate = import_module("evaluate")
except ImportError:
    evaluate = None

_bleu_metric = None


def _get_bleu_metric():
    """Lazy-load并缓存BLEU指标，避免重复初始化。"""
    global _bleu_metric
    if evaluate is None:
        raise ImportError(
            "未找到 evaluate 库，请先通过 `pip install evaluate` 安装后再启用BLEU评测。"
        )
    if _bleu_metric is None:
        _bleu_metric = evaluate.load("bleu")
    return _bleu_metric




