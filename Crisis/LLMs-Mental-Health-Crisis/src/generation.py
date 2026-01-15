import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .prompts import build_generation_prompt
from .settings import CLIENT, GENERATION_MODEL

# 日志配置
_logger = None


def _setup_logger(logs_dir: str = "outputs/logs") -> logging.Logger:
    """配置日志记录器，将日志写入文件和控制台。"""
    global _logger
    if _logger is not None:
        return _logger
    
    # 确保日志目录存在（相对于项目根目录）
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        # 如果相对路径，从当前文件位置向上找到项目根目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # src -> 项目根目录
        logs_path = project_root / logs_dir
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_path / f"generation_{timestamp}.log"
    
    # 配置日志格式
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 创建日志记录器
    logger = logging.getLogger("generation")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    _logger = logger
    logger.info(f"日志记录器已初始化，日志文件: {log_file}")
    return logger


def get_logger() -> logging.Logger:
    """获取日志记录器实例。"""
    if _logger is None:
        return _setup_logger()
    return _logger


def run_generation(
    samples: List[Dict[str, Any]],
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    logger = get_logger()
    logger.info(f"开始生成回复任务 - 样本数: {len(samples)}, 模型: {GENERATION_MODEL}, 温度: {temperature}")
    
    results: List[Dict[str, Any]] = []
    iterator = tqdm(
        samples,
        desc="生成回复",
        unit="样本",
        leave=False,
        total=len(samples),
    )
    
    error_count = 0
    for idx, item in enumerate(iterator):
        try:
            conversation_input = item.get("inputs", [])
            user_content = build_generation_prompt(conversation_input)
            
            logger.debug(f"处理样本 {idx+1}/{len(samples)} - 生成回复")
            
            completion = CLIENT.chat.completions.create(
                model=GENERATION_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": user_content},
                ],
            )
            response = completion.choices[0].message.content or ""
            
            if not response:
                logger.warning(f"样本 {idx+1} 返回空回复")
            else:
                # 记录回复长度（截断显示）
                response_preview = response[:100] + "..." if len(response) > 100 else response
                logger.debug(f"样本 {idx+1} - 生成回复长度: {len(response)} 字符, 预览: {response_preview}")
            
            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "response": response,
                }
            )
        except Exception as e:
            error_count += 1
            logger.error(f"处理样本 {idx+1} 时发生错误: {str(e)}", exc_info=True)
            # 即使出错也添加一个结果，标记为错误
            results.append(
                {
                    "inputs": item.get("inputs"),
                    "ground_truth": item.get("label"),
                    "response": f"[ERROR: {str(e)}]",
                }
            )
    
    logger.info(f"生成回复任务完成 - 总样本数: {len(samples)}, 成功: {len(samples) - error_count}, 错误: {error_count}")
    return results

