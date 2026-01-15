"""
SWMH 项目全局配置。
"""

from pathlib import Path
from openai import OpenAI

# 项目路径配置
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# API/模型配置
import os
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.org/v1")
CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 模型配置
CLASSIFICATION_MODEL = "qwen3-235b-a22b-instruct-2507"

# SWMH 数据集标签定义
# 根据实际数据集统计，SWMH 数据集包含以下5个标签（训练集分布）：
# - depression: 34.29% (11,940 samples)
# - SuicideWatch: 18.81% (6,550 samples) -> 映射为 "suicidal"
# - Anxiety: 17.62% (6,136 samples) -> 映射为 "anxiety"
# - offmychest: 15.12% (5,265 samples)
# - bipolar: 14.16% (4,932 samples)
SWMH_LABELS = {
    "depression": "抑郁症相关症状",
    "suicidal": "自杀意念或自杀相关意图（来自SuicideWatch子版块）",
    "anxiety": "焦虑症相关症状",
    "bipolar": "双相情感障碍相关症状",
    "offmychest": "一般性倾诉（来自offmychest子版块）",
}

# 标签分布统计（训练集）
SWMH_LABEL_DISTRIBUTION = {
    "depression": {"count": 11940, "percentage": 34.29},
    "suicidal": {"count": 6550, "percentage": 18.81},  # SuicideWatch
    "anxiety": {"count": 6136, "percentage": 17.62},
    "offmychest": {"count": 5265, "percentage": 15.12},
    "bipolar": {"count": 4932, "percentage": 14.16},
}

# 标签严重程度排序（用于加权评估）
SEVERITY_RANK = {
    "suicidal": 5,  # 最高优先级
    "bipolar": 4,
    "depression": 3,
    "anxiety": 2,
    "offmychest": 1,
}

