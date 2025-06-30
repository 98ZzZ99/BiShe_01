# RAG_subgraph_anomaly.py

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor           # :contentReference[oaicite:10]{index=10}
from langgraph.graph import StateGraph
from typing import Dict, Any
from RAG_tool_functions import load_data

def _detect_lof(state:Dict[str,Any])->Dict[str,Any]:    # LOF 算法实现
    df = load_data()                                    # 读原 CSV
    num_cols = ["Processing_Time","Energy_Consumption"]
    X = df[num_cols].fillna(df[num_cols].median())      # 缺失值用列中位数填充:contentReference[oaicite:9]{index=9}
    lof = LocalOutlierFactor(n_neighbors=20)            # 建模型
    outlier_flag = lof.fit_predict(X)                   # -1 = outlier, 1 = inlier
    df["is_outlier"] = (outlier_flag == -1)             # 在 DF 打标签
    state["execution_output"] = df[df["is_outlier"]]    # 仅返异常行
    return state

def build_anomaly_subgraph():
    sg = StateGraph(dict)
    sg.add_node("lof", _detect_lof)
    sg.set_entry_point("lof")
    return sg.compile()


