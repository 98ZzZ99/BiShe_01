# rag_nodes_react/execute.py

from typing import Dict, Any

def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from RAG_tool_functions import load_data
    from RAG_tools import TOOL_REGISTRY
    import pandas as pd, json, textwrap

    action = state["action"]
    fname  = action["function"]
    args   = action.get("args", {})
    cur_df = state.get("current_df")
    if cur_df is None:
        cur_df = load_data(state.get("data_path"))
    result = TOOL_REGISTRY[fname].func(cur_df, args)

    # 把 observation 写回
    if isinstance(result, pd.DataFrame):
        preview = textwrap.dedent(result.head(8).to_string(index=False))
        obs = f"[DataFrame] top-8 rows\n{preview}"
        state["current_df"] = result
    else:
        obs = f"[Scalar] {result}"
        state["last_scalar"] = result
    state["observation"] = obs
    return state


