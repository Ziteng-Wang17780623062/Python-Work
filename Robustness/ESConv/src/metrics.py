"""ESConv 生成任务评估指标计算模块。

实现 BLEU, Distinct, F1, ROUGE-L 等指标的计算。
"""

import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def ensure_nltk():
    """确保 NLTK 数据已下载到项目根目录下的 nltk_data 目录。"""
    # 获取项目根目录（ESConv/）
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # src/ -> ESConv/
    nltk_data_dir = project_root / "nltk_data"
    
    # 确保目录存在
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置 nltk 数据路径（必须使用绝对路径）
    nltk_data_path = str(nltk_data_dir.absolute())
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)
    
    # 检查并下载 punkt_tab tokenizer（新版本 nltk 需要 punkt_tab 而不是 punkt）
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print(f"正在下载 NLTK punkt_tab 数据到: {nltk_data_dir}")
        nltk.download("punkt_tab", download_dir=nltk_data_path)
        # 重新设置路径以确保能找到
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
    
    # 同时检查并下载 punkt（为了兼容性）
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print(f"正在下载 NLTK punkt 数据到: {nltk_data_dir}")
        nltk.download("punkt", download_dir=nltk_data_path)
        # 重新设置路径以确保能找到
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)


def my_lcs(string: Sequence[str], sub: Sequence[str]) -> int:
    """计算最长公共子序列（LCS）长度。"""
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0, len(sub) + 1)] for _ in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Metric:
    """评估指标计算类。"""
    
    def __init__(self, toker=None):
        self.refs: List[List[List[str]]] = []
        self.hyps: List[List[str]] = []
        self.toker = toker
        self.warnings = warnings

    def forward(self, refs: Sequence[str], hyp: str, chinese: bool = False):
        """添加一个样本的参考和生成结果。
        
        Args:
            refs: 参考文本列表（可以有多个参考）
            hyp: 生成的假设文本
            chinese: 是否为中文（影响分词方式）
        """
        if not chinese:
            self.refs.append([nltk.word_tokenize(e.lower()) for e in refs])
            self.hyps.append(nltk.word_tokenize(hyp.lower()))
        else:
            if self.toker is None:
                raise ValueError("Chinese tokenizer is required for Chinese text")
            self.refs.append([self.toker.tokenize(e) for e in refs])
            self.hyps.append(self.toker.tokenize(hyp))

    def calc_bleu_k(self, k: int) -> float:
        """计算 BLEU-k 分数。"""
        weights = [1.0 / k] * k + (4 - k) * [0.0]
        try:
            bleu = corpus_bleu(
                self.refs,
                self.hyps,
                weights=weights,
                smoothing_function=SmoothingFunction().method3,
            )
        except ZeroDivisionError:
            self.warnings.warn("the bleu is invalid")
            bleu = 0.0
        return bleu

    def calc_distinct_k(self, k: int) -> float:
        """计算 Distinct-k 分数（多样性指标）。"""
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, max(len(sen) - k + 1, 0)):
                key = tuple(sen[i : i + k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            self.warnings.warn("the distinct is invalid")
            dist = 0.0
        return dist

    def calc_unigram_f1(self):
        """计算单字 F1 分数。"""
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                cross = Counter(hyp) & Counter(ref)
                cross = sum(cross.values())
                p = cross / max(len(hyp), 1e-10)
                r = cross / max(len(ref), 1e-10)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores), f1_scores

    def calc_rouge_l(self, beta: float = 1.2):
        """计算 ROUGE-L 分数。"""
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec = []
            rec = []
            for ref in refs:
                lcs = my_lcs(ref, hyp)
                prec.append(lcs / max(len(hyp), 1e-10))
                rec.append(lcs / max(len(ref), 1e-10))
            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max != 0:
                score = ((1 + beta ** 2) * prec_max * rec_max) / float(
                    rec_max + beta ** 2 * prec_max
                )
            else:
                score = 0.0
            scores.append(score)
        return np.mean(scores), scores

    def close(self):
        """完成计算并返回所有指标。
        
        Returns:
            (整体指标字典, 单条指标字典)
        """
        result = {
            "length": float(np.mean(list(map(len, self.hyps)))) if self.hyps else 0.0,
            **{f"dist-{k}": 100 * self.calc_distinct_k(k) for k in range(1, 4)},
            **{f"bleu-{k}": 100 * self.calc_bleu_k(k) for k in range(1, 5)},
        }

        f1, scores = self.calc_unigram_f1() if self.hyps else (0.0, [])
        result["f1"] = 100 * f1
        result_list = {"f1": scores}

        rl, rouge_scores = self.calc_rouge_l() if self.hyps else (0.0, [])
        result["rouge-l"] = 100 * rl
        result_list.update({"rouge-l": rouge_scores})

        return result, result_list


def compute_metrics(
    references: List[List[str]],
    hypotheses: List[str],
    chinese: bool = False,
) -> Dict[str, any]:
    """计算所有评估指标。
    
    Args:
        references: 参考文本列表（每个样本可以有多个参考）
        hypotheses: 生成的假设文本列表
        chinese: 是否为中文
        
    Returns:
        包含所有指标的字典
    """
    ensure_nltk()
    
    metric = Metric(toker=None)
    
    # 确保 references 和 hypotheses 长度一致
    if len(references) != len(hypotheses):
        raise ValueError(
            f"References ({len(references)}) and hypotheses ({len(hypotheses)}) "
            "must have the same length"
        )
    
    # 添加所有样本
    for refs, hyp in zip(references, hypotheses):
        # 确保 refs 是列表
        if isinstance(refs, str):
            refs = [refs]
        metric.forward(refs, hyp, chinese=chinese)
    
    # 计算指标
    aggregate_metrics, per_sample_metrics = metric.close()
    
    return {
        "aggregate": aggregate_metrics,
        "per_sample": per_sample_metrics,
    }
