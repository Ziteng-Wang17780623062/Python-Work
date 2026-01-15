"""Entry point for SWMH classification evaluation."""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.classification import run_classification, score_classification
from src.data_utils import load_dataset, save_results, _sanitize_filename_part
from src.perturbation import (
    perturb_samples,
    save_perturbed_data,
    apply_perturbation,
)
from src.settings import CLASSIFICATION_MODEL, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWMH classification evaluator")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to dataset CSV (default: data/train.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results"),
        help="Directory to save results",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only first N samples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: auto-generate)")
    parser.add_argument(
        "--perturbation-type",
        type=str,
        choices=["character", "word", "sentence", "all", "none"],
        default=None,
        help="Perturbation type (character/word/sentence/all/none)",
    )
    parser.add_argument(
        "--perturbation-intensity",
        type=str,
        choices=["none", "low", "medium", "high", "all"],
        default=None,
        help="Perturbation intensity (none/low/medium/high/all)",
    )
    parser.add_argument(
        "--auto-robustness",
        action="store_true",
        help="Automatically run robustness experiments (none, low, medium, high)",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve relative path against project root."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def create_experiment_dir(
    output_dir: Path,
    model_name: str,
    perturbation_type: str,
    sample_count: int,
) -> Path:
    """创建实验文件夹，格式：模型名-扰动类别-样本数量-时间
    
    Args:
        output_dir: 基础输出目录
        model_name: 模型名称
        perturbation_type: 扰动类别
        sample_count: 样本数量
        
    Returns:
        创建的实验文件夹路径
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_sanitized = _sanitize_filename_part(model_name)
    perturbation_sanitized = _sanitize_filename_part(perturbation_type)
    
    experiment_dir_name = f"{model_sanitized}-{perturbation_sanitized}-{sample_count}-{timestamp}"
    experiment_dir = output_dir / experiment_dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def prompt_with_default(message: str, default: Optional[str] = None) -> str:
    """Prompt with optional default value."""
    prompt = f"{message}"
    if default is not None:
        prompt += f" (default: {default})"
    prompt += ": "
    user_input = input(prompt).strip()
    return user_input or (default or "")


def interactive_setup(args: argparse.Namespace) -> None:
    """交互式设置参数：先指定采样数量，然后指定扰动类别。
    
    流程：
    1. 交互指定采样数量
    2. 交互指定扰动类别
    3. 自动运行该扰动类别下的无扰动、低、中、高四种强度实验
    """
    dataset_path = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)

    print("=" * 60)
    print("SWMH 鲁棒性评估实验设置")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据集路径: {dataset_path}")
    print(f"输出目录:   {output_dir}")
    print("=" * 60)

    # 步骤1: 交互指定采样数量
    print("\n【步骤 1/2】指定采样数量")
    print("-" * 60)
    if args.limit is None:
        limit_choice = prompt_with_default(
            "请输入采样数量（输入数字，或按回车使用全部数据）",
            default="",
        )
        if limit_choice:
            try:
                limit_value = int(limit_choice)
                if limit_value <= 0:
                    raise ValueError("采样数量必须大于0")
                args.limit = limit_value
                print(f"✓ 已设置采样数量: {args.limit}")
            except ValueError as e:
                print(f"✗ 无效的采样数量，将使用全部数据: {e}")
                args.limit = None
        else:
            args.limit = None
            print("✓ 将使用全部数据")
    else:
        print(f"✓ 采样数量: {args.limit} (命令行参数)")

    # 设置随机种子（用于可重复实验）
    seed_choice = prompt_with_default(
        "请输入随机种子（用于可重复采样，留空则自动生成）",
        default="auto" if args.seed is None else str(args.seed),
    )
    if seed_choice and seed_choice.lower() not in ["auto", "none", ""]:
        try:
            args.seed = int(seed_choice)
            print(f"✓ 已设置随机种子: {args.seed}")
        except ValueError:
            print("✗ 无效的随机种子，将自动生成")
            args.seed = None
    else:
        args.seed = None
        print("✓ 将自动生成随机种子")

    # 步骤2: 交互指定扰动类别
    print("\n【步骤 2/2】指定扰动类别")
    print("-" * 60)
    print("将自动运行该扰动类别下的：无扰动、低、中、高四种强度实验")
    print()
    
    if args.perturbation_type is None:
        print("请选择扰动类别：")
        print("  1. character - 字符级扰动")
        print("  2. word - 单词级扰动")
        print("  3. sentence - 句子级扰动")
        print("  4. all - 综合扰动（字符+单词+句子）")
        print("  5. none - 无扰动（仅基线实验）")
        print()
        type_choice = prompt_with_default(
            "请输入选项 (1-5)",
            default="4",
        )
        type_map = {
            "1": "character",
            "2": "word",
            "3": "sentence",
            "4": "all",
            "5": "none",
        }
        args.perturbation_type = type_map.get(type_choice, "all")
    
    print(f"✓ 已选择扰动类别: {args.perturbation_type}")
    
    # 如果选择了"none"，只运行无扰动实验；否则运行所有强度
    if args.perturbation_type == "none":
        print("✓ 将仅运行无扰动基线实验")
        args.perturbation_intensity = "none"
    else:
        print("✓ 将自动运行：无扰动、低、中、高四种强度实验")
        args.perturbation_intensity = "all"
    
    print("\n" + "=" * 60)
    print("配置完成！")
    print("=" * 60)


def run_pipeline(
    args: argparse.Namespace,
    samples: Optional[List[Dict[str, Any]]] = None,
    perturbation_type: Optional[Literal["character", "word", "sentence", "all", "none"]] = None,
    intensity: Optional[Literal["none", "low", "medium", "high"]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run the classification evaluation pipeline and return results and filename base."""
    dataset_path = resolve_path(args.dataset)
    # output_dir 现在应该是实验文件夹，不需要再 resolve
    output_dir = args.output if isinstance(args.output, Path) else resolve_path(args.output)

    # 如果提供了samples，使用它们；否则加载数据集
    if samples is None:
        print(f"\nLoading dataset: {dataset_path}")
        samples = load_dataset(dataset_path, limit=args.limit, random_seed=args.seed)
    
    dataset_id = dataset_path.stem

    # 应用扰动（如果需要）
    if perturbation_type is None:
        perturbation_type = args.perturbation_type or "none"
    if intensity is None:
        intensity = args.perturbation_intensity or "none"

    if perturbation_type != "none" and intensity != "none":
        print(f"\n应用扰动: 类型={perturbation_type}, 强度={intensity}")
        samples = perturb_samples(samples, intensity=intensity, perturbation_type=perturbation_type)
        # 注意：扰动数据已在run_robustness_experiments中统一保存，此处不再单独保存

    print(f"\nRunning classification (samples: {len(samples)})...")
    classification_results = run_classification(
        samples, 
        dataset_id,
        perturbation_type=perturbation_type if perturbation_type != "none" else None,
        intensity=intensity if intensity != "none" else None,
    )

    # Prepare output naming
    timestamp = time.strftime("%Y%m%d_%H_%M_%S")
    model_name = _sanitize_filename_part(str(CLASSIFICATION_MODEL))
    task = "classification"
    sample_count = len(samples)
    perturbation_suffix = ""
    if perturbation_type != "none" and intensity != "none":
        perturbation_suffix = f"_{perturbation_type}_{intensity}"
    filename_base = f"{model_name}-{task}-{timestamp}-{sample_count}{perturbation_suffix}"

    print("\nScoring classification metrics...")
    classification_metrics = score_classification(
        classification_results,
        output_dir=output_dir,
        filename_base=filename_base,
        perturbation_type=perturbation_type if perturbation_type != "none" else None,
        intensity=intensity if intensity != "none" else None,
    )

    results: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "dataset_id": dataset_id,
        "task": "classification",
        "samples": len(samples),
        "model": CLASSIFICATION_MODEL,
        "perturbation_type": perturbation_type,
        "perturbation_intensity": intensity,
        "classification": classification_results,
        "classification_metrics": classification_metrics,
    }

    return results, filename_base


def save_summary_report(
    all_results: List[Dict[str, Any]],
    intensities: List[str],
    perturbation_type: str,
    output_dir: Path,
    dataset_path: Path,
) -> Tuple[Path, Path]:
    """生成并保存汇总报告文件。
    
    Args:
        all_results: 所有实验结果列表
        intensities: 强度列表
        perturbation_type: 扰动类型
        output_dir: 输出目录
        dataset_path: 数据集路径
        
    Returns:
        (JSON汇总文件路径, 文本汇总文件路径)
    """
    import json
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_stem = dataset_path.stem
    model_name = _sanitize_filename_part(str(CLASSIFICATION_MODEL))
    
    # 生成汇总数据
    summary_data = {
        "experiment_info": {
            "dataset": str(dataset_path),
            "dataset_id": dataset_stem,
            "model": CLASSIFICATION_MODEL,
            "perturbation_type": perturbation_type,
            "timestamp": timestamp,
            "total_experiments": len(all_results),
        },
        "results": [],
        "summary_metrics": {},
    }
    
    # 收集所有结果
    summary_metrics = {
        "accuracy": {},
        "precision_macro": {},
        "recall_macro": {},
        "macro_f1": {},
        "micro_f1": {},
    }
    
    for i, result in enumerate(all_results):
        intensity = intensities[i]
        metrics = result.get("classification_metrics", {})
        
        result_entry = {
            "intensity": intensity,
            "perturbation_type": perturbation_type if intensity != "none" else "none",
            "samples": result.get("samples", 0),
            "metrics": {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision_macro": metrics.get("precision_macro", 0.0),
                "recall_macro": metrics.get("recall_macro", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "micro_f1": metrics.get("micro_f1", 0.0),
            },
            "classification_report": metrics.get("classification_report", {}),
            "confusion_matrix": metrics.get("confusion_matrix", {}),
        }
        summary_data["results"].append(result_entry)
        
        # 收集汇总指标
        summary_metrics["accuracy"][intensity] = metrics.get("accuracy", 0.0)
        summary_metrics["precision_macro"][intensity] = metrics.get("precision_macro", 0.0)
        summary_metrics["recall_macro"][intensity] = metrics.get("recall_macro", 0.0)
        summary_metrics["macro_f1"][intensity] = metrics.get("macro_f1", 0.0)
        summary_metrics["micro_f1"][intensity] = metrics.get("micro_f1", 0.0)
    
    summary_data["summary_metrics"] = summary_metrics
    
    # 保存JSON格式汇总
    summary_filename = f"{model_name}_robustness_summary_{perturbation_type}_{timestamp}.json"
    summary_path = output_dir / summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 生成可读的文本汇总报告
    txt_summary = []
    txt_summary.append("=" * 80)
    txt_summary.append("SWMH 鲁棒性评估实验汇总报告")
    txt_summary.append("=" * 80)
    txt_summary.append("")
    txt_summary.append(f"数据集: {dataset_path}")
    txt_summary.append(f"模型: {CLASSIFICATION_MODEL}")
    txt_summary.append(f"扰动类型: {perturbation_type}")
    txt_summary.append(f"实验时间: {timestamp}")
    txt_summary.append(f"实验数量: {len(all_results)}")
    txt_summary.append("")
    txt_summary.append("-" * 80)
    txt_summary.append("整体指标汇总")
    txt_summary.append("-" * 80)
    txt_summary.append("")
    txt_summary.append(f"{'强度':<12} {'准确率':<12} {'精确率(宏)':<14} {'召回率(宏)':<14} {'F1(宏)':<12} {'F1(微)':<12}")
    txt_summary.append("-" * 80)
    
    for i, result in enumerate(all_results):
        intensity = intensities[i]
        metrics = result.get("classification_metrics", {})
        txt_summary.append(
            f"{intensity:<12} "
            f"{metrics.get('accuracy', 0.0):<12.4f} "
            f"{metrics.get('precision_macro', 0.0):<14.4f} "
            f"{metrics.get('recall_macro', 0.0):<14.4f} "
            f"{metrics.get('macro_f1', 0.0):<12.4f} "
            f"{metrics.get('micro_f1', 0.0):<12.4f}"
        )
    
    txt_summary.append("")
    txt_summary.append("-" * 80)
    txt_summary.append("详细结果")
    txt_summary.append("-" * 80)
    txt_summary.append("")
    
    for i, result in enumerate(all_results):
        intensity = intensities[i]
        metrics = result.get("classification_metrics", {})
        txt_summary.append(f"\n【{intensity.upper()} 强度】")
        txt_summary.append(f"样本数: {result.get('samples', 0)}")
        txt_summary.append(f"准确率: {metrics.get('accuracy', 0.0):.4f}")
        txt_summary.append(f"精确率(宏): {metrics.get('precision_macro', 0.0):.4f}")
        txt_summary.append(f"召回率(宏): {metrics.get('recall_macro', 0.0):.4f}")
        txt_summary.append(f"F1分数(宏): {metrics.get('macro_f1', 0.0):.4f}")
        txt_summary.append(f"F1分数(微): {metrics.get('micro_f1', 0.0):.4f}")
        
        # 每个类别的详细指标
        classification_report = metrics.get("classification_report", {})
        if classification_report:
            txt_summary.append("\n各类别详细指标:")
            txt_summary.append(f"{'类别':<15} {'精确率':<12} {'召回率':<12} {'F1':<12} {'支持数':<10}")
            txt_summary.append("-" * 70)
            for label in sorted(classification_report.keys()):
                if label in ["macro avg", "micro avg", "weighted avg"]:
                    continue
                report = classification_report[label]
                txt_summary.append(
                    f"{label:<15} "
                    f"{report.get('precision', 0.0):<12.4f} "
                    f"{report.get('recall', 0.0):<12.4f} "
                    f"{report.get('f1-score', 0.0):<12.4f} "
                    f"{report.get('support', 0):<10}"
                )
        txt_summary.append("")
    
    txt_summary.append("=" * 80)
    txt_summary.append("报告结束")
    txt_summary.append("=" * 80)
    
    # 保存文本汇总
    txt_summary_path = output_dir / f"{model_name}_robustness_summary_{perturbation_type}_{timestamp}.txt"
    with txt_summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(txt_summary))
    
    return summary_path, txt_summary_path


def run_robustness_experiments(args: argparse.Namespace) -> None:
    """自动运行鲁棒性实验（无扰动、低、中、高四种强度）。"""
    dataset_path = resolve_path(args.dataset)
    base_output_dir = resolve_path(args.output)

    print("\n" + "=" * 60)
    print("开始自动鲁棒性实验")
    print("=" * 60)

    # 加载数据集（只加载一次，确保所有实验使用相同的数据）
    print(f"\n加载数据集: {dataset_path}")
    original_samples = load_dataset(dataset_path, limit=args.limit, random_seed=args.seed)
    dataset_id = dataset_path.stem
    print(f"✓ 已加载 {len(original_samples)} 个样本")

    # 确定扰动类型
    perturbation_type = args.perturbation_type or "all"
    
    # 创建实验文件夹
    experiment_dir = create_experiment_dir(
        base_output_dir,
        CLASSIFICATION_MODEL,
        perturbation_type,
        len(original_samples),
    )
    print(f"\n✓ 创建实验文件夹: {experiment_dir}")
    
    # 如果选择了"none"，只运行无扰动实验
    if perturbation_type == "none":
        intensities = ["none"]
    else:
        # 运行四种强度的实验
        intensities = ["none", "low", "medium", "high"]
    
    all_results = []
    all_result_files = []

    # 先对所有强度进行扰动并分别保存
    perturbed_data_files = {}
    perturbed_data_dir = experiment_dir / "perturbed_data"
    perturbed_data_dir.mkdir(parents=True, exist_ok=True)
    # 时间戳格式：YYYYMMDD-HHMMSS（使用连字符分隔）
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    print("\n" + "=" * 60)
    print("步骤1: 应用扰动并保存数据")
    print("=" * 60)
    
    for intensity in intensities:
        print(f"\n处理强度: {intensity.upper()}")
        
        # 对于无扰动，使用原始数据；否则应用扰动
        if intensity == "none":
            samples = original_samples.copy()
            print("使用原始数据（无扰动）")
        else:
            print(f"应用扰动: 类型={perturbation_type}, 强度={intensity}")
            samples = perturb_samples(
                original_samples.copy(),
                intensity=intensity,
                perturbation_type=perturbation_type,
            )
        
        # 保存扰动后的数据为CSV格式（文件命名：时间-扰动类别-强度）
        if intensity == "none":
            perturbed_filename = f"{timestamp}-none-none.csv"
        else:
            perturbed_filename = f"{timestamp}-{perturbation_type}-{intensity}.csv"
        perturbed_path = perturbed_data_dir / perturbed_filename
        save_perturbed_data(samples, perturbed_path)
        perturbed_data_files[intensity] = perturbed_path
        print(f"✓ 扰动数据已保存: {perturbed_path}")
    
    # 步骤2: 使用保存的扰动数据进行实验
    print("\n" + "=" * 60)
    print("步骤2: 使用保存的扰动数据进行实验")
    print("=" * 60)
    
    for intensity in intensities:
        print("\n" + "=" * 60)
        print(f"运行实验 [{intensity.upper()}]: 扰动类型={perturbation_type}")
        print("=" * 60)
        
        # 从保存的CSV文件加载扰动后的数据
        perturbed_csv_path = perturbed_data_files[intensity]
        print(f"从文件加载数据: {perturbed_csv_path}")
        # 从已保存的CSV文件加载，不需要再次采样，使用全部数据
        samples = load_dataset(perturbed_csv_path, limit=None, random_seed=None)
        print(f"✓ 已加载 {len(samples)} 个样本")

        # 运行分类评估
        results, filename_base = run_pipeline(
            args,
            samples=samples,
            perturbation_type=perturbation_type if intensity != "none" else "none",
            intensity=intensity,
        )
        all_results.append(results)
        
        # 保存单个实验结果
        result_path = save_results(results, experiment_dir, filename_base=filename_base)
        all_result_files.append(result_path)
        print(f"✓ 实验结果已保存: {result_path}")

    # 生成并保存汇总报告
    print("\n" + "=" * 60)
    print("生成汇总报告")
    print("=" * 60)
    summary_json_path, summary_txt_path = save_summary_report(
        all_results,
        intensities,
        perturbation_type,
        experiment_dir,
        dataset_path,
    )
    print(f"✓ 汇总报告(JSON)已保存: {summary_json_path}")
    print(f"✓ 汇总报告(文本)已保存: {summary_txt_path}")

    # 打印汇总结果
    print("\n" + "=" * 60)
    print("鲁棒性实验汇总")
    print("=" * 60)
    print(f"{'强度':<12} {'准确率':<12} {'精确率(宏)':<14} {'召回率(宏)':<14} {'F1(宏)':<12} {'F1(微)':<12}")
    print("-" * 80)
    for i, result in enumerate(all_results):
        intensity = intensities[i]
        metrics = result.get("classification_metrics", {})
        print(
            f"{intensity:<12} "
            f"{metrics.get('accuracy', 0.0):<12.4f} "
            f"{metrics.get('precision_macro', 0.0):<14.4f} "
            f"{metrics.get('recall_macro', 0.0):<14.4f} "
            f"{metrics.get('macro_f1', 0.0):<12.4f} "
            f"{metrics.get('micro_f1', 0.0):<12.4f}"
        )

    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)
    print(f"\n汇总报告:")
    print(f"  JSON格式: {summary_json_path}")
    print(f"  文本格式: {summary_txt_path}")
    print(f"\n所有实验结果文件:")
    for i, file_path in enumerate(all_result_files):
        print(f"  {i+1}. {file_path}")


def main() -> None:
    args = parse_args()
    interactive_setup(args)

    # 根据配置决定运行模式
    # 如果扰动类型为"none"，只运行基线实验
    # 否则，自动运行无扰动、低、中、高四种强度的实验
    if args.perturbation_type == "none":
        # 只运行无扰动基线实验
        print("\n运行无扰动基线实验...")
        
        # 先加载数据集以获取样本数量
        dataset_path = resolve_path(args.dataset)
        samples = load_dataset(dataset_path, limit=args.limit, random_seed=args.seed)
        
        # 创建实验文件夹
        base_output_dir = resolve_path(args.output)
        experiment_dir = create_experiment_dir(
            base_output_dir,
            CLASSIFICATION_MODEL,
            "none",
            len(samples),
        )
        print(f"\n✓ 创建实验文件夹: {experiment_dir}")
        
        # 修改 args.output 为实验文件夹
        args.output = experiment_dir
        
        results, filename_base = run_pipeline(
            args,
            samples=samples,
            perturbation_type="none",
            intensity="none",
        )
        output_path = save_results(results, experiment_dir, filename_base=filename_base)
        print(f"\n✓ 实验结果已保存: {output_path}")
    else:
        # 自动运行所有强度的鲁棒性实验
        run_robustness_experiments(args)


if __name__ == "__main__":
    main()

