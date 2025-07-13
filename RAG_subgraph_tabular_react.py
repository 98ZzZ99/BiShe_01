# RAG_subgraph_tabular_react.py

from langgraph.graph import StateGraph, END
from rag_nodes_react.thought    import thought_node
from rag_nodes_react.validator  import validator_node
from rag_nodes_react.execute    import execute_node     # 就是上面函数

# 接收一份 dict（当前 state），返回下一个节点的名字 ("execute" / "thought" / END)
def _validate_switch(state: dict) -> str:
    """按照 state['route'] 把 validate 派发到下一个节点"""
    route = state.get("route")
    if route == "execute":
        return "execute"
    if route == "finish":
        return END          # 0.4.8 里仍然用 END 常量
    # "error" 或 其他
    return "thought"

# 1. 用 StateGraph(dict) 创建空图
# 2. 注册 3 个节点 + 边
# 3. 把 _validate_switch 塞进 add_conditional_edges
# 4. 设入口 → compile()

def build_tabular_react_subgraph():
    sg = StateGraph(dict)
    sg.add_node("thought", thought_node)
    sg.add_node("validate", validator_node)
    sg.add_node("execute",  execute_node)

    sg.add_edge("thought", "validate")
    # sg.add_edge("execute",  "validate")  # ← 把原来的 execute→thought 改到 validate
    sg.add_conditional_edges(
        "execute",
        lambda s: "validate" if s.get("route") in {"thought", "error", "execute"} else END
    )

    # ★ 一行就够：把路径函数丢进去
    sg.add_conditional_edges("validate", _validate_switch)

    sg.set_entry_point("thought")
    return sg.compile()

