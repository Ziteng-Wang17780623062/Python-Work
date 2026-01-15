"""
谄媚性评估项目全局配置。
"""

from pathlib import Path
from openai import OpenAI

# 项目路径配置
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
USED_DATA_DIR = PROJECT_ROOT / "outputs" / "used_data"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

# API/模型配置
import os
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 模型配置
TEST_MODEL = "gpt-4o-mini"  # 被测试的LLM模型
JUDGE_MODEL = "gpt-4o"  # 裁判LLM模型

# 数据集配置
DATASETS = {
    "OEQ": {
        "path": DATA_DIR / "OEQ.csv",
        "name": "OEQ",
        "prompt_column": "prompt",  # 用户问题所在的列
    },
    "AITA-YTA": {
        "path": DATA_DIR / "AITA-YTA.csv",
        "name": "AITA-YTA",
        "prompt_column": "prompt",  # 用户问题所在的列
    },
    "PAS": {
        "path": DATA_DIR / "PAS.csv",
        "name": "PAS",
        "prompt_column": "text",  # 用户问题所在的列（需要确认）
    },
}

