# RAG_subgraph_viz.py

import matplotlib.pyplot as plt
from RAG_tool_functions import load_data
from langgraph.graph import StateGraph
from typing import Dict, Any
import os

def _simple_viz(state:Dict[str,Any])->Dict[str,Any]:
    df = load_data()                            # 读取全表
    out = "output/processing_time_hist.png"     # 固定输出路径
    plt.figure()                                # 打开新画布
    df["Processing_Time"].hist()                # 画直方图
    plt.title("Processing Time histogram")
    os.makedirs("output", exist_ok=True)  # 确保输出目录存在
    plt.savefig(out)                            # :contentReference[oaicite:11]{index=11}
    plt.close()                                 # 释放内存
    state["execution_output"] = out             # 把文件路径写回 state
    return state

def build_viz_subgraph():
    sg = StateGraph(dict)
    sg.add_node("viz", _simple_viz)
    sg.set_entry_point("viz")
    return sg.compile()


