# rag_nodes_react/execute.py

from RAG_tool_functions import load_data
from RAG_tools import TOOL_REGISTRY
import pandas as pd, json, textwrap
from typing import Dict, Any

MAX_PREVIEW = 10        # 打给用户的 DataFrame 行数

def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:

    # 1) 从 state 中取出 action 和参数
    action = state["action"]        # action 是 validator 用 Pydantic 校验过的 Action 模型实例；它有两个字段：function:str 和 args:dict
    fname  = action["function"]     # == "filter_date_range"，拿到用户(或 LLM) 希望调用的工具名
    args   = action.get("args", {}) # == {"column": "Scheduled_Start", "value": "15:00"}，拿到调用时需要的参数字典
    # cur_df = state.get("current_df")
    cur_df = state.get("execution_output")  # 上一次结果

    # 2) 检查工具名是否在注册表中，调用工具
    try:
        result = TOOL_REGISTRY[fname].func(cur_df, args)
    except Exception as e:
        # 工具报错，让validator_node 知道
        state["observation"] = f"[Tool-Error] {e}"
        state["route"] = "error"    # 错误，回到thought
        return state

    # 3) 根据结果类型更新 state（即共享状态）
    # 把最新 DF/标量放进共享区，供后续 Action 使用
    state["execution_output"] = result
    if isinstance(result, pd.DataFrame):
        preview = textwrap.dedent(result.head(MAX_PREVIEW).to_string(index=False))
        state["final_answer"] = f"[DataFrame] top-{MAX_PREVIEW} rows\n{preview}"
    else:
        state["final_answer"] = str(result)

    state["route"] = "finish"            # 关键信号：子图结束

    # # 本 demo 直接把结果当最终答案
    # if hasattr(result, "head"):
    #     answer = result.head(MAX_PREVIEW).to_string(index=False)
    # else:
    #     answer = str(result)
    #
    # state["final_answer"] = answer
    # state["route"] = "finish"  # 告诉子图：结束

    # if cur_df is None:
    #     cur_df = load_data(state.get("data_path"))
    # result = TOOL_REGISTRY[fname].func(cur_df, args)    # 先从字典取出对应工具，再访问 .func 得到它原始的 Python 函数并真正执行。
    #
    # # 把 observation 写回
    # if isinstance(result, pd.DataFrame):
    #     preview = textwrap.dedent(result.head(8).to_string(index=False))
    #     obs = f"[DataFrame] top-8 rows\n{preview}"
    #     state["current_df"] = result
    # else:
    #     obs = f"[Scalar] {result}"
    #     state["last_scalar"] = result
    # state["observation"] = obs

    print("[DEBUG] execute result type:", type(result))
    print("[DEBUG] final_answer preview:", str(state.get("final_answer"))[:120])

    return state


