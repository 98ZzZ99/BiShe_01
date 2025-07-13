# rag_nodes_react/execute.py

from __future__ import annotations
import textwrap, pandas as pd, logging
from typing import Dict, Any
from RAG_tools import TOOL_REGISTRY

log = logging.getLogger("rag.execute")

MAX_PREVIEW = 10                # DataFrame 打印行数
SIDE_EFFECT_FUNCS = {
    "add_derived_column",       # 直接修改文件或磁盘
    "graph_export",
    "plot_machine_avg_bar",
}

def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    依 action_queue 逐条执行工具。
    • result 若为 DF/标量写入 execution_output
    • route = thought (还有后续) / finish (已结束)
    """

    cur = state.get("execution_output")

    queue = state.get("action_queue", [])
    log.debug("ENTER execute_node | pending-queue=%s", queue)

    if not queue:                       # 全部执行完
        state.update(route="finish",
                     final_answer=str(cur))             # 结果转成字符串或自定义渲染
        log.debug("No more actions, route -> finish")
        return state

    # action = queue.pop(0)               # 取当前 action
    action = state["action_queue"].pop(0)
    fname = action["function"]
    args  = action.get("args", {})
    state["action_queue"] = queue       # 写回剩余

    log.debug("Run tool %s | args=%s | cur_type=%s",
              fname, args, type(cur).__name__)

    try:
        logging.debug("Run tool %s | args=%s | cur_type=%s", fname, args, type(cur).__name__)
        result = TOOL_REGISTRY[fname].func(cur, args)
        logging.debug("Tool ok | result_type=%s", type(result).__name__)

    except Exception as e:
        state["observation"] = f"[Tool-Error] {fname} {args} -> {e}"
        state["route"] = "error"
        log.exception("Tool raised exception:")
        return state

    # ---- 更新共享状态 -------------------------------------------------
    state["execution_output"] = result
    if isinstance(result, pd.DataFrame):
        preview = textwrap.dedent(result.head(MAX_PREVIEW).to_string(index=False))
        state["final_answer"] = f"[DataFrame] top-{MAX_PREVIEW} rows\n{preview}"
    else:
        state["final_answer"] = str(result)

    log.debug("Tool ok | result_type=%s | route_decision=%s",
              type(result).__name__,
              "finish" if fname in SIDE_EFFECT_FUNCS or not queue else "thought")

    # ---- 结束 / 继续 --------------------------------------------------
    state["route"] = "finish" if (fname in SIDE_EFFECT_FUNCS or not queue) else "thought"
    if queue:                       # 队列里还有 action
        state["route"] = "execute"  # 继续执行下一步，不再回 thought/validator
    else:                           # 队列已空
        state["route"] = "finish"

    return state
