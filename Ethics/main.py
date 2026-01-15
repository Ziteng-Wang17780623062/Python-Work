"""EthicsMH 论理性测试实验入口。"""

from pathlib import Path

from src.pipeline import run_reasoning_evaluation
from src.settings import (
    DATASET_PATH,
    EVALUATION_MODEL,
    DEFAULT_TEMPERATURE,
)


def _ask_sample_size() -> int:
    """交互式获取采样数量，确保为正整数。"""
    while True:
        raw = input("请输入本次采样数量（正整数）：").strip()
        try:
            value = int(raw)
        except ValueError:
            print("无效输入，请输入数字。")
            continue
        if value <= 0:
            print("采样数量必须大于 0。")
            continue
        return value


def main():
    print("=== EthicsMH 论理性测试 ===")
    print(f"默认数据集：{DATASET_PATH.name}")
    print(f"默认模型：{EVALUATION_MODEL}")
    sample_size = _ask_sample_size()

    result = run_reasoning_evaluation(
        dataset_path=Path(DATASET_PATH),
        sample_size=sample_size,
        model=EVALUATION_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        random_seed=None,  # 传入 None，内部自动生成随机种子，保证每次不同
    )

    saved = result.get("saved_path")
    if saved:
        print(f"本次结果已保存：{saved}")
    else:
        print("结果保存路径未返回，请检查日志。")


if __name__ == "__main__":
    main()