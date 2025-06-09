#  RAG_graph_config.py

"""Builds a 3-node LangGraph pipeline: preprocess → prompt → execute."""
from typing import TypedDict, Any, Dict
from langgraph.graph import StateGraph, END
from RAG_node_0_preprocessing import PreprocessingNode
from RAG_node_router          import RouterNode
from RAG_subgraph_tabular     import build_tabular_subgraph
from RAG_subgraph_kg      import build_kg_subgraph
from RAG_subgraph_anomaly import build_anomaly_subgraph
from RAG_subgraph_viz     import build_viz_subgraph

class PipelineState(TypedDict): # TypedDict用于给 dict 加上「键必须存在且类型固定」的静态约束。这相当于在类型层面声明：我们的「共享状态」一定包含这四个键。
    user_input:         str
    processed_input:    str
    route:              str # <─ new  by Router
    llm_json:           str
    execution_output:   Any # DataFrame / scalar / file path / str
# 调用 StateGraph(PipelineState) 时，LangGraph 会用这份类型信息来做静态检查：每个节点对 state 的读写都应该遵守这个 schema。这样能在开发期提前暴露字段拼写或类型错误。

# 以下三个并不是类，而是普通包裹函数，每个函数内部真正调用的类实例就是等号右边的部分。
_Pre  = PreprocessingNode()
_Router = RouterNode()

# ---------- wrapper functions --------------------------------------------
def _pre(state:Dict[str,Any])->Dict[str,Any]:
    state["processed_input"] = _Pre.run(state["user_input"])
    return state

def _route(state:Dict[str,Any])->Dict[str,Any]:
    state.update(_Router.run(state["processed_input"]))
    return state

# ---------- build graph ---------------------------------------------------
def build_graph():
    sg = StateGraph(PipelineState)

    # root line
    sg.add_node("pre", _pre) # "pre" 是节点名；_preprocess 为执行函数。运行时 Graph 把 state 交给该函数
    sg.add_node("router", _route)

    sg.add_edge("pre", "router")

    # sub-graphs
    tabular = build_tabular_subgraph()  # 返回编译好的 StateGraph
    sg.add_subgraph("tabular", tabular) # add_subgraph 方法会把子图的所有节点和边添加到主图中（把它当成父图的“巨型节点”）。
    sg.add_subgraph("kg", build_kg_subgraph())
    sg.add_subgraph("anomaly", build_anomaly_subgraph())
    sg.add_subgraph("viz", build_viz_subgraph())

    # router → conditional edges
    sg.add_conditional_edges(   # conditional_edges 把起点 "router" 按条件派发到多目的节点，即router核心功能。
        "router",
        [
            (lambda s: s["route"]=="kg",      "kg"),
            (lambda s: s["route"]=="anomaly", "anomaly"),
            (lambda s: s["route"]=="viz",     "viz"),
        ],
        # default branch
        default = "tabular"
    )

    # all branches converge to END
    for node in ("tabular","kg","anomaly","viz"):
        sg.add_edge(node, END)

    sg.set_entry_point("pre")
    return sg.compile()


