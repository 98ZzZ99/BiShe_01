# RAG_node_0_preprocessing.py

"""
预处理节点：
1. 读取用户可能给出的 csv 路径（或默认 data 文件夹）并抓取列名；
2. 利用 RapidFuzz 与 Sentence-Transformers 计算列名相似度；
3. 把用户输入中的“近似列名”替换成真实列名；
4. 返回纠正后的英文查询（供 Router 与下游使用）。
"""
from __future__ import annotations
import re, os, contextlib
from typing import List
import pandas as pd
from rapidfuzz import process, fuzz          # pip install rapidfuzz
from sentence_transformers import SentenceTransformer, util  # pip install sentence-transformers
from RAG_tool_functions import load_data
from pathlib import Path

# SentenceTransformer 是 Hugging-Face 上的句向量库，可把字符串映射到 384 维向量，做语义相似度计算。
# all-MiniLM-L6-v2 是官方提供的轻量 22 M 参数模型，速度是 mpnet-base 的 5 倍而保留 ~90 % 质量，非常适合实时纠错任务。
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")   # 体积小、加载快

def _find_csv(candidate: str | None) -> Path | None:
    """
    1) 若 candidate 是绝对/相对路径且存在 → 返回
    2) 否则依次在以下目录查找：
       • data/
       • data/A/
       • 项目根目录
    找到即返回 Path；都找不到返回 None
    """
    search_dirs = [Path.cwd() / "data",
                   Path.cwd() / "data" / "A",
                   Path.cwd()]
    if candidate:
        p = Path(candidate)
        if p.is_file():
            return p
        # 只给了文件名，尝试在 search_dirs 里拼路径
        search_dirs = [d for d in search_dirs]

    for d in search_dirs:
        p = d / candidate if candidate else None
        if p and p.is_file():
            return p
    return None
# ---------------------------------------------------------------- #

class PreprocessingNode:
    def __init__(self) -> None:
        print("[LOG] PreprocessingNode (robust) initialized.")

    def run(self, user_input: str) -> str:
        print("[LOG] Preprocessing running …")

        # 1. 捕获 *.csv
        m = re.search(r"\b([A-Za-z0-9_./\\-]+\.csv)\b", user_input)
        csv_path = _find_csv(m.group(1)) if m else None
        if csv_path is None:
            # 完全找不到 → 抛错误，交由上层捕获
            raise FileNotFoundError("❌ 找不到任何 CSV 文件，请检查文件名/路径")

        df = load_data(str(csv_path))            # ← 一定能成功
        columns = df.columns.tolist()

        # 2. 列名纠错（同原逻辑，略）
        def _replace(match):
            word = match.group(0)
            real = _best_match(word, columns)
            return real or word

        pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
        corrected = pattern.sub(_replace, user_input)
        corrected = re.sub(r"\s+", " ", corrected).strip()

        print("[BEFORE]", user_input)
        print("[AFTER ]", corrected)

        # ---- 关键改动 ----
        return {
            "processed_input": corrected,
            "csv_path": str(csv_path)  # 让后续子图能直接拿到路径
        }

def _best_match(token:str, columns:List[str])->str|None:
    # 形参 token：     在用户文本里捕获的「可能是列名」的词
    # 形参 columns：   CSV 中的真实列名列表
    # 返回值 str | None 表示「找到匹配就给列名字符串，否则返回 None」
    """先用 RapidFuzz 筛出 top-3，再用向量相似度择优"""
    # 1) 传统编辑距离
    # 先把 token 与每个列名算相似度，保留得分 > 60 的 Top-3。WRatio 是综合 Levenshtein/部分匹配等指标的加权分。
    candidates = [c for c,score,_ in process.extract(token, columns, scorer=fuzz.WRatio, limit=3) if score > 80]
    if not candidates:
        return None
    # 2) 语义向量相似度
    # 再用 SBERT 将 token 和候选列名编码成向量，用余弦相似度二次排名，最后返回分数最高列名。
    emb_token = _MODEL.encode(token, convert_to_tensor=True)
    emb_cols  = _MODEL.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(emb_token, emb_cols)[0].tolist()
    return candidates[scores.index(max(scores))]


