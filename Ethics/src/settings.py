"""
EthicsMH 论理性测试实验的基础配置。

参考 C-SSRS 项目的配置方式，集中管理模型、客户端与默认路径。
"""

from pathlib import Path

from openai import OpenAI

# 项目根目录（EthicsMH）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据与输出目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "Ethical-Reasoning-in-Mental-Health.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


# API 客户端，从环境变量读取配置
import os
CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
)

# 被测大模型名称（如需更换，请直接修改本文件）
EVALUATION_MODEL = "qwen3-235b-a22b-instruct-2507"

# 裁判模型名称（LLM-as-judge）
JUDGE_MODEL = "gpt-4o-mini"

# 默认温度
DEFAULT_TEMPERATURE = 0.0


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DATASET_PATH",
    "OUTPUT_DIR",
    "LOG_DIR",
    "CLIENT",
    "EVALUATION_MODEL",
    "DEFAULT_TEMPERATURE",
]
