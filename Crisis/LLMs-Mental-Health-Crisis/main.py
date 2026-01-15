import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from src.classification import run_classification, score_classification
from src.data_utils import load_dataset, read_protocol, save_results
from src.generation import run_generation
from src.judging import evaluate_generation_outputs
from src.settings import CLASSIFICATION_MODEL, GENERATION_MODEL
from src.visualization import plot_confusion_matrix_from_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="心理健康分类/生成任务评估器")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/human_label/human-labeled-sampled_dataset_n206_s42-merged_labels.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("data/protocol.csv"),
        help="包含标签描述的 protocol.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results"),
        help="结果保存目录",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "generation", "all"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅取前 N 条样本")
    return parser.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path: Path) -> Path:
    """将相对路径解析为相对于项目根目录的绝对路径。"""
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate


def prompt_with_default(message: str, default: Optional[str] = None) -> str:
    """带默认值的输入提示。"""
    prompt = f"{message}"
    if default is not None:
        prompt += f" (默认: {default})"
    prompt += ": "
    user_input = input(prompt).strip()
    return user_input or (default or "")


def interactive_setup(args: argparse.Namespace) -> None:
    """在运行前展示配置信息并让用户选择任务和样本数量。"""
    dataset_path = resolve_path(args.dataset)
    protocol_path = resolve_path(args.protocol)
    output_dir = resolve_path(args.output)

    print("========== 运行配置 ==========")
    print(f"数据集路径: {dataset_path}")
    print(f"协议路径:   {protocol_path}")
    print(f"输出目录:   {output_dir}")
    print(f"默认任务:   {args.task}")
    print(
        f"默认样本数: {args.limit if args.limit is not None else '全部'}"
    )
    print("================================")

    task_options = {
        "1": "classification",
        "2": "generation",
        "3": "all",
    }
    task_choice = prompt_with_default(
        "请选择测试类别 [1:分类 2:生成 3:全部]",
        default="3",
    )
    if task_choice in task_options:
        args.task = task_options[task_choice]
    elif task_choice in task_options.values():
        args.task = task_choice
    else:
        print("输入无效，保持默认任务。")

    limit_choice = prompt_with_default(
        "请输入测试样本数量 (空值代表全部)",
        default=str(args.limit) if args.limit is not None else "",
    )
    if limit_choice:
        try:
            parsed_limit = int(limit_choice)
            if parsed_limit <= 0:
                raise ValueError
            args.limit = parsed_limit
        except ValueError:
            print("样本数量输入无效，保持默认值。")


def _determine_model_label(task: str) -> str:
    models = []
    if task in {"classification", "all"}:
        models.append(CLASSIFICATION_MODEL)
    if task in {"generation", "all"}:
        models.append(GENERATION_MODEL)
    if not models:
        return "unknown_model"
    # 去重保持顺序
    seen = []
    for model in models:
        if model not in seen:
            seen.append(model)
    return "+".join(seen)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = resolve_path(args.dataset)
    protocol_path = resolve_path(args.protocol)
    output_dir = resolve_path(args.output)

    original_dataset_path = dataset_path
    original_samples, random_seed = load_dataset(dataset_path, args.limit)
    dataset_id = original_dataset_path.stem

    # 在正式跑模型前，输出本次采样到的数据中包含的标签类别信息
    labels_in_sample = sorted(
        {
            str(item.get("label")).strip()
            for item in original_samples
            if item.get("label") is not None and str(item.get("label")).strip()
        }
    )
    print("========== 采样标签概览 ==========")
    print(f"样本数量: {len(original_samples)}")
    print(f"包含的标签种类数: {len(labels_in_sample)}")
    print(f"标签列表: {labels_in_sample}")
    print("================================")
    
    # Load protocol scales for evaluation
    print(f"[协议] 正在从 {protocol_path} 加载评估协议...")
    try:
        _, label_scales = read_protocol(protocol_path)
        print(f"[协议] 协议加载成功，包含 {len(label_scales)} 个标签的评估标准")
    except FileNotFoundError as e:
        print(f"[错误] 协议文件未找到: {e}")
        raise
    except ValueError as e:
        print(f"[错误] 协议文件无效: {e}")
        raise

    results: Dict[str, Any] = {
        "dataset": str(original_dataset_path),
        "protocol": str(args.protocol),
        "task": args.task,
        "samples": len(original_samples),
        "model": _determine_model_label(args.task),
        "sampled_labels": labels_in_sample,
    }
    
    # 如果进行了采样，记录使用的随机种子
    if random_seed is not None:
        results["random_seed"] = random_seed
        results["sampling_info"] = {
            "random_seed": random_seed,
            "sampled": True,
            "sample_count": len(original_samples),
        }

    if args.task in {"classification", "all"}:
        classification_results = run_classification(original_samples, dataset_id)
        classification_metrics = score_classification(
            classification_results,
        )
        results["classification"] = classification_results
        results["classification_metrics"] = classification_metrics

    if args.task in {"generation", "all"}:
        generation_results = run_generation(original_samples)
        generation_judgment = evaluate_generation_outputs(
            generation_results,
            label_scales,
        )
        results["generation"] = generation_results
        results["generation_judgment"] = generation_judgment

    return results


def main() -> None:
    args = parse_args()
    interactive_setup(args)
    results = run_pipeline(args)
    output_path = save_results(results, resolve_path(args.output))
    print(f"结果已保存至: {output_path}")
    
    # Generate confusion matrix visualization if classification metrics are available
    if "classification_metrics" in results:
        try:
            print("Generating confusion matrix...")
            plot_confusion_matrix_from_json(
                output_path,
                output_path.parent,
                normalize=False,
            )
            # Also generate normalized version
            plot_confusion_matrix_from_json(
                output_path,
                output_path.parent,
                normalize=True,
            )
            print("Confusion matrix visualization completed!")
        except ImportError as e:
            print(f"Warning: Cannot generate visualization. Please ensure matplotlib and seaborn are installed: {e}")
        except Exception as e:
            print(f"Warning: Error occurred while generating confusion matrix: {e}")


if __name__ == "__main__":
    main()
