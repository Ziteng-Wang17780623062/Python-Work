"""
谄媚性评估实验主程序。
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_utils import (
    load_dataset,
    save_sampled_data,
    update_sampled_data_with_responses,
    update_sampled_data_with_judgments,
)
from src.generation import generate_responses_for_samples
from src.judging import judge_responses_for_samples
from src.results import save_judge_results
from src.settings import DATASETS, PROJECT_ROOT, TEST_MODEL


def prompt_with_default(message: str, default: Optional[str] = None) -> str:
    """提示用户输入，支持默认值。"""
    prompt = f"{message}"
    if default is not None:
        prompt += f" (默认: {default})"
    prompt += ": "
    user_input = input(prompt).strip()
    return user_input or (default or "")


def select_dataset() -> str:
    """交互式选择数据集。"""
    print("=" * 60)
    print("谄媚性评估实验")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print("=" * 60)
    
    print("\n【步骤 1/2】选择数据集")
    print("-" * 60)
    print("请选择要测试的数据集：")
    print("  1. OEQ")
    print("  2. AITA-YTA")
    print("  3. PAS")
    print()
    
    while True:
        choice = prompt_with_default("请输入选项 (1-3)", default="1")
        dataset_map = {
            "1": "OEQ",
            "2": "AITA-YTA",
            "3": "PAS",
        }
        if choice in dataset_map:
            dataset_name = dataset_map[choice]
            print(f"✓ 已选择数据集: {dataset_name}")
            return dataset_name
        else:
            print("✗ 无效选项，请输入 1、2 或 3")


def select_sample_limit() -> Optional[int]:
    """交互式选择采样数量。"""
    print("\n【步骤 2/2】指定采样数量")
    print("-" * 60)
    
    while True:
        limit_choice = prompt_with_default(
            "请输入采样数量（输入数字，或按回车使用全部数据）",
            default="",
        )
        if not limit_choice:
            print("✓ 将使用全部数据")
            return None
        
        try:
            limit_value = int(limit_choice)
            if limit_value <= 0:
                raise ValueError("采样数量必须大于0")
            print(f"✓ 已设置采样数量: {limit_value}")
            return limit_value
        except ValueError as e:
            print(f"✗ 无效的采样数量: {e}")
            print("请重新输入")


def run_experiment(dataset_name: str, sample_limit: Optional[int]) -> None:
    """运行实验流程。"""
    print("\n" + "=" * 60)
    print("开始实验")
    print("=" * 60)
    
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # 步骤1: 加载数据集并采样
    print("\n【步骤 1/4】加载数据集并采样")
    print("-" * 60)
    samples, random_seed = load_dataset(
        dataset_name=dataset_name,
        limit=sample_limit,
        random_seed=None  # 自动生成随机种子，确保每次不同
    )
    print(f"✓ 已加载 {len(samples)} 个样本 (随机种子: {random_seed})")
    
    # 步骤2: 保存采样数据
    print("\n【步骤 2/4】保存采样数据")
    print("-" * 60)
    sampled_data_path = save_sampled_data(dataset_name, samples, timestamp)
    print(f"✓ 采样数据已保存")
    
    # 步骤3: 使用被测试LLM自由生成回复
    # 测试逻辑：使用数据集中的prompt，让被测试的LLM自由生成回复
    print("\n【步骤 3/4】使用被测试LLM自由生成回复")
    print("-" * 60)
    print("正在使用数据集中的prompt让被测试LLM自由生成回复...")
    responses = generate_responses_for_samples(samples)
    print(f"✓ 已生成 {len(responses)} 个回复")
    
    # 更新采样数据文件，添加回复列
    update_sampled_data_with_responses(sampled_data_path, responses)
    print(f"✓ 回复已保存到数据文件")
    
    # 步骤4: 使用裁判LLM进行裁决（输出0/1）
    # 测试逻辑：使用judge LLM对回复进行评估，输出0/1
    print("\n【步骤 4/4】使用裁判LLM进行裁决（输出0/1）")
    print("-" * 60)
    print("正在进行裁决（0=不谄媚，1=谄媚）...")
    # 使用默认的裁判提示词（已在judging.py中配置）
    judge_prompt = None  # 使用默认的JUDGE_PROMPT_TEMPLATE
    judgments = judge_responses_for_samples(samples, responses, judge_prompt=judge_prompt)
    print(f"✓ 已完成 {len(judgments)} 个裁决（输出0/1）")
    
    # 更新采样数据文件，添加裁决列
    update_sampled_data_with_judgments(sampled_data_path, judgments)
    print(f"✓ 裁决已保存到数据文件")
    
    # 步骤5: 保存结果汇总
    print("\n【步骤 5/5】保存结果汇总")
    print("-" * 60)
    # 使用settings中配置的被测试模型名称
    model_name = TEST_MODEL or "未配置模型"
    results_path = save_judge_results(
        dataset_name=dataset_name,
        model_name=model_name,
        judgments=judgments,
        samples=samples,
        timestamp=timestamp
    )
    print(f"✓ 结果汇总已保存")
    
    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)
    print(f"采样数据文件: {sampled_data_path}")
    print(f"结果汇总文件: {results_path}")
    print("=" * 60)


def main() -> None:
    """主函数。"""
    try:
        # 交互选择数据集
        dataset_name = select_dataset()
        
        # 交互选择采样数量
        sample_limit = select_sample_limit()
        
        # 运行实验
        run_experiment(dataset_name, sample_limit)
        
    except KeyboardInterrupt:
        print("\n\n实验已取消")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

