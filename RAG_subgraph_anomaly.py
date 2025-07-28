# RAG_subgraph_anomaly.py
"""
Anomaly‑subgraph —— Extended Isolation Forest (EIF) via `isotree`
------------------------------------------------------------------
• 入口节点 'eif'：
      ① 读取 preprocessing 写入的 csv_path（若无则用默认）
      ② 只取数值列 → EIF 打分
      ③ 结果写入 execution_output 供主图打印
"""

from typing import Dict, Any, TypedDict
import time, pandas as pd, numpy as np
import os
from pathlib import Path
from rag_eval import run_evaluation
from sklearn.metrics import average_precision_score
from langgraph.graph import StateGraph
from RAG_tool_functions import load_data        # 你已有
from rag_algorithms import ALGOS, run_algo
from rag_eval import evaluate

# ────────── 1. 计算节点 ──────────
def _benchmark(state: dict, top_q: float = 0.02) -> dict:
    """
    对所有算法跑分，记录 latency 与 ‘自评’ PR‑AUC，
    把得分最高模型的 anomaly_score 写回 DataFrame 并导出 Excel。
    """
    csv_path: str = state["csv_path"]            # 预处理节点填好的
    df        = pd.read_csv(csv_path)
    X         = df.drop(columns=["time_stamp"]).values   # 只用数值列

    rows   = []          # 每行记录一个算法的汇总
    scores_map = {}      # 保存每个算法的分数数组，稍后选最优

    for name in ALGOS.keys():
        t0 = time.perf_counter()
        scores = run_algo(name, X)               # ← 你的算法跑分
        dt = time.perf_counter() - t0

        # “自评”：用本模型自己 top‑2% 作为 pseudo label
        thr = np.quantile(scores, 1 - top_q)  # 取 98% 分位数
        y_ref = (scores >= thr).astype(int)  # 置 1 = 异常
        pr_auc = average_precision_score(y_ref, scores)

        print("阈值:", thr)
        print("正样本个数 (≈20):", y_ref.sum())
        print("自评 PR‑AUC:", pr_auc)

        rows.append({
            "algo":    name,
            "seconds": round(dt, 2),
            "pr_auc":  round(pr_auc, 4),
        })
        scores_map[name] = scores

    bench_df = pd.DataFrame(rows)

    # —— 选 pr_auc 最高的模型 ——
    best = bench_df.sort_values("pr_auc", ascending=False)["algo"].iloc[0]

    # 把最佳模型的分数写入原 DataFrame
    df["anomaly_score"] = scores_map[best]

    # 保存 Excel（每个算法一个工作表 + benchmark 汇总）
    excel_path = os.path.splitext(csv_path)[0] + "_anomaly_results.xlsx"
    # with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        # 每个模型的分数
        # for algo, sc in scores_map.items():
        #     tmp = df[["time_stamp"]].copy()
        #     tmp["anomaly_score"] = sc
        #     tmp.to_excel(w, sheet_name=algo, index=False)
        # 保存每个算法的逐行分数。
        # 统一所有工作表的列名为 time_stamp + anomaly_score，
        # rag_eval.py 在读取时就不会再因找不到列而抛 KeyError: 'anomaly_score'。
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as w:
        # --- 保存每个算法的逐行分数 ---
        for algo, sc in scores_map.items():  # ← 这里用 algo
            algo_df = pd.DataFrame({
                "time_stamp": df["time_stamp"].astype(str),
                "anomaly_score": sc
            })
            algo_df.to_excel(w, sheet_name=algo, index=False)  # ← sheet_name=algo

        # 摘要
        bench_df.to_excel(w, sheet_name="benchmark", index=False)

    # ——— 把结果写回 state ———
    state["execution_output"] = df.sort_values("anomaly_score",
                                               ascending=False).head(5)
    state["bench_summary"]    = bench_df
    state["picked_algo"]      = best
    state["excel_path"]       = excel_path
    return state

# -------- 子图 state 描述 --------
class AnomalyState(TypedDict, total=False):
    # 这几项是 _benchmark 已写进去的
    execution_output: Any          # DataFrame (供 printer 打印 Top‑5)
    bench_summary:    Any          # DataFrame (各算法评估表)
    picked_algo:      str
    excel_path:       str

    # 新增的评估汇总
    eval_summary:     Any

    # 其余字段（父图会用到）留空也可以
    processed_input:  str
    csv_path:         str | None

# -------- 在 benchmark 之后跑评估 --------
def _post_eval(state: dict) -> dict:
    """
    打开 benchmark 写出的 Excel，再跑一次评估+可视化，
    并把结果写回同一个 state 供 printer 使用。
    """
    excel_path = state["excel_path"]
    summary = run_evaluation(excel_path)      # rag_eval.py 中的主入口
    state["eval_summary"] = summary           # 打印 & 调试用
    return state

# -------- 组装子图 --------
def build_anomaly_subgraph():
    sg = StateGraph(AnomalyState)

    sg.add_node("benchmark", _benchmark)   # ← 你已有的节点
    sg.add_node("post_eval", _post_eval)   # ← 新增节点

    sg.add_edge("benchmark", "post_eval")  # benchmark → post_eval

    sg.set_entry_point("benchmark")
    sg.set_finish_point("post_eval")       # 子图终点 = post_eval

    return sg.compile()

