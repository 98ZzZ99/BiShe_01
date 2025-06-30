# RAG_subgraph_tabular_react.py

from langgraph.graph import StateGraph, END
from rag_nodes_react.thought    import thought_node
from rag_nodes_react.validator  import validator_node
from rag_nodes_react.execute    import execute_node     # 就是上面函数

# def flatten_cases(cases):
#     flat = []
#     for cond, dest in cases:
#         flat.extend([cond, dest])   # 依次压入 cond、dest
#     return flat

# def build_tabular_react_subgraph():
#     sg = StateGraph(dict)
#
#     sg.add_node("thought", thought_node)
#     sg.add_node("validate", validator_node)
#     sg.add_node("execute", execute_node)
#
#     sg.add_edge("thought", "validate")
#
#     cases = [
#         (lambda s: s.get("route") == "execute", "execute"),
#         (lambda s: s.get("route") == "error", "thought"),
#         (lambda s: s.get("route") == "finish", END),
#     ]
#
#     flat = [item for pair in cases for item in pair]  # 打平。 cond1,dest1,cond2,dest2,cond3,dest3
#
#     # 把 fallback 放在最后一个位置参数
#     sg.add_conditional_edges("validate", *flat, "thought")  # ← 没有 default=
#
#     sg.add_edge("execute", "thought")
#     sg.set_entry_point("thought")
#     return sg.compile()

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
    sg.add_edge("execute", "thought")

    # ★ 一行就够：把路径函数丢进去
    sg.add_conditional_edges("validate", _validate_switch)

    sg.set_entry_point("thought")
    return sg.compile()

