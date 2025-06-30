# RAG_node_0_preprocessing.py

# class PreprocessingNode:
#     def __init__(self) -> None:
#         print("[LOG] PreprocessingNode initialized.")
#
#     def run(self, user_input: str) -> str:
#         print("[LOG] PreprocessingNode running …")
#         out = user_input.strip()
#         print(f"[LOG] Preprocessing completed. Output: {out}")
#         return out

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

# SentenceTransformer 是 Hugging-Face 上的句向量库，可把字符串映射到 384 维向量，做语义相似度计算。
# all-MiniLM-L6-v2 是官方提供的轻量 22 M 参数模型，速度是 mpnet-base 的 5 倍而保留 ~90 % 质量，非常适合实时纠错任务。
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")   # 体积小、加载快

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

class PreprocessingNode:
    # 抓真实列名：load_data() 读 CSV→df.columns。
    # 列名语义纠错：执行上文算法，把模糊列名改成真列名，降低下游报错。
    # 日志：__init__ 只打印一次，run() 打印纠正前后文本，便于调试。

    def __init__(self) -> None:
        print("[LOG] PreprocessingNode (with column-name mapping) initialized.")

    def run(self, user_input:str)->str:
        print("[LOG] Preprocessing running …")

        # —— 1. 尝试解析用户输入里的 csv 路径 ——
        m = re.search(r"\b([A-Za-z0-9_./\\-]+\.csv)\b", user_input)
        with contextlib.suppress(Exception):
            df = load_data(m.group(1)) if m else load_data()
        columns = df.columns.tolist()

        # —— 2. 全词扫描并纠正近似列名 ——
        def _replace(match):
            # _replace() 回调把每个匹配词送入 _best_match()，若成功找到真实列名则替换，否则保留原词。
            word = match.group(0)
            real = _best_match(word, columns)
            return real or word

        pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")   # 类 SQL 列标识符
        # 抓取形似 SQL 标识符的词（字母/下划线开头，后续可含数字）。
        # \b 是「单词边界」，确保只匹配完整文件名 xxx.csv 而非子串。
        corrected = pattern.sub(_replace, user_input)

        # —— 3. 去除多余空格，统一大小写 ——
        corrected = re.sub(r"\s+", " ", corrected).strip()

        print("[BEFORE]", user_input)
        print("[AFTER ]", corrected)

        print(f"[LOG] → corrected query: {corrected}")
        return corrected

