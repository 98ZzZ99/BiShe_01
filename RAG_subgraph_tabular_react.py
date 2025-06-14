# RAG_subgraph_tabular_react.py

from langgraph.graph import StateGraph
from rag_nodes_react.thought    import thought_node
from rag_nodes_react.validator  import validator_node
from rag_nodes_react.execute    import execute_node     # 就是上面函数

def build_tabular_react_subgraph():
    sg = StateGraph(dict)

    sg.add_node("thought", thought_node)
    sg.add_node("validate", validator_node)
    sg.add_node("execute", execute_node)

    sg.add_edge("thought", "validate")

    # 条件边
    sg.add_conditional_edges(
        "validate",
        [
            (lambda s: s.get("route")=="execute", "execute"),
            (lambda s: s.get("route")=="error",   "thought"),
            (lambda s: s.get("route")=="finish",  END),
        ],
        default="thought",
    )

    sg.add_edge("execute", "thought")

    sg.set_entry_point("thought")
    return sg.compile()


