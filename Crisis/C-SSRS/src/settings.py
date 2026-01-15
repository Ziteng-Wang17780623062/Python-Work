"""
C-SSRS 项目全局配置。
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
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 模型配置
CLASSIFICATION_MODEL = "deepseek-v3.1-250821"

# C-SSRS 标签定义（基于论文的5标签分类）
# 根据论文，标签严重程度从低到高：
# Supportive < Indicator < Ideation < Behavior < Attempt
C_SSRS_LABELS = {
    "Supportive": "支持性回复：提供支持、鼓励和帮助的回复内容",
    "Indicator": "自杀指标：使用风险语言但主要表达同情和支持，而非个人自杀意图",
    "Ideation": "自杀意念：存在自杀想法但尚未采取具体行动",
    "Behavior": "自杀行为：已采取准备性行为或表达明确的自杀意图",
    "Attempt": "自杀尝试：已实际执行自杀行为，无论是否成功",
}

# 标签严重程度排序（用于计算Graded指标）
# 数值越大，严重程度越高
SEVERITY_RANK = {
    "Supportive": 0,
    "Indicator": 1,
    "Ideation": 2,
    "Behavior": 3,
    "Attempt": 4,
}

# 标签反向映射：从数值到标签
SEVERITY_TO_LABEL = {v: k for k, v in SEVERITY_RANK.items()}

