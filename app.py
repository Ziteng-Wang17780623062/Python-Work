"""
后端API服务器，用于调用各个测评模块
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time

app = Flask(__name__)
CORS(app)

# 确保JSON响应使用UTF-8编码
@app.after_request
def after_request(response):
    """确保所有响应都使用UTF-8编码"""
    if response.content_type and 'application/json' in response.content_type:
        response.charset = 'utf-8'
    return response

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 各个模块的路径
MODULES = {
    "crisis_llms": BASE_DIR / "危机识别与升级" / "LLMs-Mental-Health-Crisis",
    "crisis_cssrs": BASE_DIR / "危机识别与升级" / "C-SSRS",
    "ethics": BASE_DIR / "伦理性",
    "sycophancy": BASE_DIR / "谄媚性",
    "robustness_esconv": BASE_DIR / "鲁棒性" / "ESConv",
    "robustness_swmh": BASE_DIR / "鲁棒性" / "SWMH",
}

# 存储任务状态（使用文件系统持久化）
TASK_STATUS_FILE = BASE_DIR / "task_status.json"
task_status = {}

def load_task_status():
    """从文件加载任务状态"""
    global task_status
    if TASK_STATUS_FILE.exists():
        try:
            with open(TASK_STATUS_FILE, "r", encoding="utf-8") as f:
                task_status = json.load(f)
        except Exception as e:
            print(f"加载任务状态失败: {e}")
            task_status = {}

def save_task_status():
    """保存任务状态到文件"""
    try:
        # 创建一个可序列化的副本
        serializable_status = {}
        for task_id, status in task_status.items():
            serializable_status[task_id] = {}
            for key, value in status.items():
                if key in ["start_time", "end_time"] and isinstance(value, float):
                    serializable_status[task_id][key] = value
                elif key == "result" and isinstance(value, dict):
                    # 确保result字典可以序列化
                    serializable_status[task_id][key] = {
                        "returncode": value.get("returncode", 0),
                        "stdout": value.get("stdout", ""),
                        "stderr": value.get("stderr", "")
                    }
                else:
                    serializable_status[task_id][key] = value
        
        with open(TASK_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_status, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存任务状态失败: {e}")

# 启动时加载任务状态
load_task_status()


def run_crisis_llms(params):
    """运行LLMs-Mental-Health-Crisis模块 - 直接调用，不使用子进程"""
    import sys
    from pathlib import Path
    from io import StringIO
    
    module_dir = MODULES["crisis_llms"]
    
    # 将模块路径添加到sys.path
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    
    # 重定向stdout和stderr以捕获输出
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        # 直接导入并调用模块函数
        from src.classification import run_classification, score_classification
        from src.data_utils import load_dataset, read_protocol, save_results
        from src.generation import run_generation
        from src.judging import evaluate_generation_outputs
        from src.settings import CLASSIFICATION_MODEL, GENERATION_MODEL
        from src.visualization import plot_confusion_matrix_from_json
        
        PROJECT_ROOT = Path(module_dir)
        dataset_path = PROJECT_ROOT / "data" / "human_label" / "human-labeled-sampled_dataset_n206_s42-merged_labels.json"
        protocol_path = PROJECT_ROOT / "data" / "protocol.csv"
        output_dir = PROJECT_ROOT / "outputs" / "results"
        
        task = params.get("task", "all")
        limit = params.get("limit")
        
        print("========== 开始执行任务 ==========")
        print(f"任务类型: {task}")
        print(f"样本限制: {limit if limit else '无限制'}")
        print("=================================")
        
        # 加载数据
        original_samples, random_seed = load_dataset(dataset_path, limit)
        dataset_id = dataset_path.stem
        
        labels_in_sample = sorted({
            str(item.get("label")).strip()
            for item in original_samples
            if item.get("label") is not None and str(item.get("label")).strip()
        })
        
        print("========== 采样标签概览 ==========")
        print(f"样本数量: {len(original_samples)}")
        print(f"包含的标签种类数: {len(labels_in_sample)}")
        print(f"标签列表: {labels_in_sample}")
        print("================================")
        
        # 加载协议
        print(f"[协议] 正在从 {protocol_path} 加载评估协议...")
        try:
            _, label_scales = read_protocol(protocol_path)
            print(f"[协议] 协议加载成功，包含 {len(label_scales)} 个标签的评估标准")
        except Exception as e:
            print(f"[错误] 协议加载失败: {e}")
            raise
        
        # 确定模型标签
        models = []
        if task in {"classification", "all"}:
            models.append(CLASSIFICATION_MODEL)
        if task in {"generation", "all"}:
            models.append(GENERATION_MODEL)
        model_label = "+".join(models) if models else "unknown_model"
        
        results = {
            "dataset": str(dataset_path),
            "protocol": str(protocol_path),
            "task": task,
            "samples": len(original_samples),
            "model": model_label,
            "sampled_labels": labels_in_sample,
        }
        
        if random_seed is not None:
            results["random_seed"] = random_seed
            results["sampling_info"] = {
                "random_seed": random_seed,
                "sampled": True,
                "sample_count": len(original_samples),
            }
        
        # 执行分类任务
        if task in {"classification", "all"}:
            print("\n运行分类任务...")
            classification_results = run_classification(original_samples, dataset_id)
            classification_metrics = score_classification(classification_results)
            results["classification"] = classification_results
            results["classification_metrics"] = classification_metrics
        
        # 执行生成任务
        if task in {"generation", "all"}:
            print("\n运行生成任务...")
            generation_results = run_generation(original_samples)
            generation_judgment = evaluate_generation_outputs(
                generation_results,
                label_scales,
            )
            results["generation"] = generation_results
            results["generation_judgment"] = generation_judgment
        
        # 保存结果
        output_path = save_results(results, output_dir)
        print(f"结果已保存至: {output_path}")
        
        # 生成混淆矩阵
        if "classification_metrics" in results:
            try:
                # 设置matplotlib使用非交互式后端，避免在非主线程中使用GUI的警告
                import matplotlib
                matplotlib.use('Agg')  # 使用非交互式后端
                
                print("Generating confusion matrix...")
                plot_confusion_matrix_from_json(
                    output_path,
                    output_path.parent,
                    normalize=False,
                )
                plot_confusion_matrix_from_json(
                    output_path,
                    output_path.parent,
                    normalize=True,
                )
                print("Confusion matrix visualization completed!")
            except Exception as e:
                print(f"Warning: Error occurred while generating confusion matrix: {e}")
        
        print("\n========== 任务执行完成 ==========")
        
        # 获取输出
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        return {
            "returncode": 0,
            "stdout": stdout,
            "stderr": stderr
        }
        
    except Exception as e:
        stderr = stderr_capture.getvalue() + f"\n执行异常: {str(e)}"
        import traceback
        stderr += f"\n{traceback.format_exc()}"
        return {
            "returncode": 1,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr
        }
    finally:
        # 恢复stdout和stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # 从sys.path中移除模块路径（可选）
        if str(module_dir) in sys.path:
            sys.path.remove(str(module_dir))


def run_crisis_cssrs(params):
    """运行C-SSRS模块"""
    module_dir = MODULES["crisis_cssrs"]
    
    limit_value = params.get("limit") if params.get("limit") else "None"
    
    # 直接调用核心函数，避免交互式输入
    script_content = f"""
# -*- coding: utf-8 -*-
import sys
import io
# 强制使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import time
from pathlib import Path
sys.path.insert(0, r"{module_dir}")

from src.classification import run_classification, score_classification
from src.data_utils import load_dataset, save_results
from src.settings import CLASSIFICATION_MODEL, PROJECT_ROOT
from src.visualization import plot_confusion_matrix_from_json

dataset_path = PROJECT_ROOT / "data" / "500_Reddit_users_posts_labels.csv"
output_dir = PROJECT_ROOT / "outputs" / "results"

sample_limit = {limit_value}
random_seed = int(time.time() * 1000000) % (2**32)

print("=" * 80)
print("C-SSRS Classification Task Evaluator")
print("=" * 80)
print(f"Dataset Path: {{dataset_path}}")
print(f"Output Directory: {{output_dir}}")
print(f"Model: {{CLASSIFICATION_MODEL}}")
print(f"Sample Limit: {{sample_limit}}")
print(f"Random Seed: {{random_seed}}")
print("=" * 80)
print()

print(f"Loading dataset: {{dataset_path}}")
samples = load_dataset(dataset_path, limit=sample_limit, random_seed=random_seed)
dataset_id = dataset_path.stem
print(f"Loaded {{len(samples)}} samples")
print()

print("=" * 80)
print("Running classification task...")
print("=" * 80)
classification_results = run_classification(samples, dataset_id)

print()
print("=" * 80)
print("Calculating all evaluation metrics...")
print("=" * 80)
classification_metrics = score_classification(classification_results)

results = {{
    "dataset": str(dataset_path),
    "dataset_id": dataset_id,
    "task": "classification",
    "samples": len(samples),
    "sample_limit": sample_limit,
    "random_seed": random_seed,
    "model": CLASSIFICATION_MODEL,
    "classification": classification_results,
    "metrics": classification_metrics,
}}

print()
print("=" * 80)
print("Saving results...")
print("=" * 80)
output_path = save_results(results, output_dir)
print(f"\\nAll results saved to: {{output_path}}")

if "metrics" in results:
    try:
        print()
        print("=" * 80)
        print("Generating confusion matrix visualization...")
        print("=" * 80)
        plot_confusion_matrix_from_json(
            output_path,
            output_path.parent,
            normalize=False,
        )
        plot_confusion_matrix_from_json(
            output_path,
            output_path.parent,
            normalize=True,
        )
        print("Confusion matrix visualization completed!")
    except Exception as e:
        print(f"Warning: Error occurred while generating confusion matrix: {{e}}")

print()
print("=" * 80)
print("Evaluation completed!")
print("=" * 80)
"""
    temp_script = module_dir / "temp_run.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置环境变量，强制使用UTF-8编码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, str(temp_script)],
        cwd=str(module_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    stdout, stderr = process.communicate()
    
    if temp_script.exists():
        temp_script.unlink()
    
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr
    }


def run_ethics(params):
    """运行伦理性模块"""
    module_dir = MODULES["ethics"]
    
    sample_size = params.get("sample_size", 10)
    
    script_content = f"""
# -*- coding: utf-8 -*-
import sys
import io
# 强制使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r"{module_dir}")
from main import *
from src.pipeline import run_reasoning_evaluation
from pathlib import Path
from src.settings import DATASET_PATH, EVALUATION_MODEL, DEFAULT_TEMPERATURE

sample_size = {sample_size}
result = run_reasoning_evaluation(
    dataset_path=Path(DATASET_PATH),
    sample_size=sample_size,
    model=EVALUATION_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    random_seed=None,
)
print(f"结果已保存: {{result.get('saved_path', '未知')}}")
"""
    temp_script = module_dir / "temp_run.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置环境变量，强制使用UTF-8编码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, str(temp_script)],
        cwd=str(module_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    stdout, stderr = process.communicate()
    
    if temp_script.exists():
        temp_script.unlink()
    
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr
    }


def run_sycophancy(params):
    """运行谄媚性模块"""
    module_dir = MODULES["sycophancy"]
    
    dataset_name = params.get("dataset", "OEQ")
    sample_limit = params.get("limit", None)
    
    script_content = f"""
# -*- coding: utf-8 -*-
import sys
import io
# 强制使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r"{module_dir}")
from main import run_experiment
import time

dataset_name = "{dataset_name}"
sample_limit = {sample_limit if sample_limit else "None"}

run_experiment(dataset_name, sample_limit)
"""
    temp_script = module_dir / "temp_run.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置环境变量，强制使用UTF-8编码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, str(temp_script)],
        cwd=str(module_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    stdout, stderr = process.communicate()
    
    if temp_script.exists():
        temp_script.unlink()
    
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr
    }


def run_robustness_esconv(params):
    """运行ESConv鲁棒性模块"""
    module_dir = MODULES["robustness_esconv"]
    
    limit = params.get("limit", None)
    perturbation_type = params.get("perturbation_type", "all")
    seed = params.get("seed", None)
    
    script_content = f"""
# -*- coding: utf-8 -*-
import sys
import io
# 强制使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r"{module_dir}")
from main import *
import argparse

args = argparse.Namespace()
args.dataset = Path("data/ESConv.json")
args.output = Path("outputs/results")
args.limit = {limit if limit else "None"}
args.seed = {seed if seed else "None"}
args.context_window = 6
args.per_dialog_limit = 2
args.temperature = 0.0
args.max_tokens = 512
args.perturbation_type = "{perturbation_type}"
args.perturbation_intensity = "all" if "{perturbation_type}" != "none" else "none"
args.auto_robustness = False

if args.perturbation_type == "none":
    # 只运行无扰动基线实验
    dataset_path = resolve_path(args.dataset)
    samples = build_samples_from_dataset(
        dataset_path,
        limit=args.limit,
        context_window=args.context_window,
        per_dialog_limit=args.per_dialog_limit,
        random_seed=args.seed,
    )
    base_output_dir = resolve_path(args.output)
    experiment_dir = create_experiment_dir(
        base_output_dir,
        GENERATION_MODEL,
        "none",
        len(samples),
    )
    args.output = experiment_dir
    results, filename_base = run_pipeline(
        args,
        samples=samples,
        perturbation_type="none",
        intensity="none",
    )
    output_path = save_results(results, experiment_dir, filename_base=filename_base)
    print(f"实验结果已保存: {{output_path}}")
else:
    run_robustness_experiments(args)
"""
    temp_script = module_dir / "temp_run.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置环境变量，强制使用UTF-8编码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, str(temp_script)],
        cwd=str(module_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    stdout, stderr = process.communicate()
    
    if temp_script.exists():
        temp_script.unlink()
    
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr
    }


def run_robustness_swmh(params):
    """运行SWMH鲁棒性模块"""
    module_dir = MODULES["robustness_swmh"]
    
    limit = params.get("limit", None)
    perturbation_type = params.get("perturbation_type", "all")
    seed = params.get("seed", None)
    
    script_content = f"""
# -*- coding: utf-8 -*-
import sys
import io
# 强制使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r"{module_dir}")
from main import *
import argparse

args = argparse.Namespace()
args.dataset = Path("data/train.csv")
args.output = Path("outputs/results")
args.limit = {limit if limit else "None"}
args.seed = {seed if seed else "None"}
args.perturbation_type = "{perturbation_type}"
args.perturbation_intensity = "all" if "{perturbation_type}" != "none" else "none"
args.auto_robustness = False

if args.perturbation_type == "none":
    dataset_path = resolve_path(args.dataset)
    samples = load_dataset(dataset_path, limit=args.limit, random_seed=args.seed)
    base_output_dir = resolve_path(args.output)
    experiment_dir = create_experiment_dir(
        base_output_dir,
        CLASSIFICATION_MODEL,
        "none",
        len(samples),
    )
    args.output = experiment_dir
    results, filename_base = run_pipeline(
        args,
        samples=samples,
        perturbation_type="none",
        intensity="none",
    )
    output_path = save_results(results, experiment_dir, filename_base=filename_base)
    print(f"实验结果已保存: {{output_path}}")
else:
    run_robustness_experiments(args)
"""
    temp_script = module_dir / "temp_run.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置环境变量，强制使用UTF-8编码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, str(temp_script)],
        cwd=str(module_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    stdout, stderr = process.communicate()
    
    if temp_script.exists():
        temp_script.unlink()
    
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr
    }


@app.route("/")
def index():
    """返回主页面"""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/crisis-llms")
def crisis_llms():
    """返回危机识别与升级 (LLMs) 页面"""
    return send_from_directory(BASE_DIR, "crisis_llms.html")


@app.route("/crisis-cssrs")
def crisis_cssrs():
    """返回危机识别与升级 (C-SSRS) 页面"""
    return send_from_directory(BASE_DIR, "crisis_cssrs.html")


@app.route("/ethics")
def ethics():
    """返回伦理性测评页面"""
    return send_from_directory(BASE_DIR, "ethics.html")


@app.route("/sycophancy")
def sycophancy():
    """返回谄媚性测评页面"""
    return send_from_directory(BASE_DIR, "sycophancy.html")


@app.route("/robustness-esconv")
def robustness_esconv():
    """返回鲁棒性测评 (ESConv) 页面"""
    return send_from_directory(BASE_DIR, "robustness_esconv.html")


@app.route("/robustness-swmh")
def robustness_swmh():
    """返回鲁棒性测评 (SWMH) 页面"""
    return send_from_directory(BASE_DIR, "robustness_swmh.html")


@app.route("/api/run", methods=["POST"])
def run_task():
    """运行测评任务"""
    data = request.json
    module = data.get("module")
    params = data.get("params", {})
    
    if module not in ["crisis_llms", "crisis_cssrs", "ethics", "sycophancy", "robustness_esconv", "robustness_swmh"]:
        return jsonify({"error": "无效的模块名称"}), 400
    
    # 生成任务ID
    task_id = f"{module}_{int(time.time() * 1000)}"
    task_status[task_id] = {
        "status": "running",
        "module": module,
        "start_time": time.time()
    }
    save_task_status()  # 保存状态
    
    def run_in_thread():
        try:
            if module == "crisis_llms":
                result = run_crisis_llms(params)
            elif module == "crisis_cssrs":
                result = run_crisis_cssrs(params)
            elif module == "ethics":
                result = run_ethics(params)
            elif module == "sycophancy":
                result = run_sycophancy(params)
            elif module == "robustness_esconv":
                result = run_robustness_esconv(params)
            elif module == "robustness_swmh":
                result = run_robustness_swmh(params)
            else:
                result = {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "未知的模块名称"
                }
            
            task_status[task_id]["status"] = "completed" if result.get("returncode", 1) == 0 else "failed"
            task_status[task_id]["result"] = result
            task_status[task_id]["end_time"] = time.time()
            save_task_status()  # 保存状态
        except Exception as e:
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["error"] = str(e)
            task_status[task_id]["result"] = {
                "returncode": 1,
                "stdout": "",
                "stderr": f"执行异常: {str(e)}"
            }
            task_status[task_id]["end_time"] = time.time()
            save_task_status()  # 保存状态
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    
    return jsonify({"task_id": task_id, "status": "running"})


@app.route("/api/status/<task_id>", methods=["GET"])
def get_status(task_id):
    """获取任务状态"""
    # 重新加载任务状态（防止Flask重启导致状态丢失）
    load_task_status()
    
    if task_id not in task_status:
        return jsonify({"error": "任务不存在"}), 404
    
    # 重新加载任务状态（防止Flask重启导致状态丢失）
    load_task_status()
    
    if task_id not in task_status:
        return jsonify({"error": "任务不存在"}), 404
    
    status = task_status[task_id].copy()
    if "result" in status and status["result"] is not None:
        # 限制输出大小
        if "stdout" in status["result"] and status["result"]["stdout"] is not None:
            status["result"]["stdout"] = status["result"]["stdout"][-5000:]  # 只返回最后5000字符
        if "stderr" in status["result"] and status["result"]["stderr"] is not None:
            status["result"]["stderr"] = status["result"]["stderr"][-5000:]
    
    return jsonify(status)


@app.route("/<path:filename>")
def static_files(filename):
    """返回静态文件（CSS、JS等）"""
    # 排除API路由
    if filename.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

