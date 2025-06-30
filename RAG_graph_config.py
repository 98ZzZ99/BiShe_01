#  RAG_graph_config.py

"""Builds a 3-node LangGraph pipeline: preprocess → prompt → execute."""
from typing import TypedDict, Any, Dict
from langgraph.graph import StateGraph, END

from RAG_node_0_preprocessing import PreprocessingNode
from RAG_node_router          import RouterNode
from RAG_subgraph_anomaly import build_anomaly_subgraph
from RAG_subgraph_tabular_react import build_tabular_react_subgraph

# ---------- sub-graph adapters ----------

def passthrough(subgraph):
    """父、子 state 完全兼容时：直接当 node 用"""
    """passthrough 直接把编译好的子图对象当作节点返回。"""
    return subgraph          # 直接返回，无额外适配

def wrap_with_mapping(subgraph, to_child, from_child):
    """
    当父/子 schema 部分不同：
      to_child(parent_state)  -> child_state
      from_child(parent_state, child_state) -> parent_state
    也就是说，当子图所需 state 与父图 不完全相同 时，需要「进、出」各做一次转换。
      schema ＝ 这份 state 里有哪些键、类型、意义。这里我们说 “schema 不同”，等价于“字段不兼容”。
    """
    def _runner(state):
        child_in  = to_child(state.copy())      # state.copy()：防止在 to_child() 里不小心修改原字典。to_child，经典字典操作，输入父图 state，返回子图能接受的 state（删/改/补字段）
        child_out = subgraph.invoke(child_in)   # subgraph.invoke(child_in)，LangGraph 提供的同步执行接口，把格式已对的 child_in 送进子图内部所有节点依次执行，产生新的 child_state (child_out)
        return from_child(state, child_out)     # from_child：一个函数/λ表达式，输入父-state 和子-state，返回转换后的父-state。也就是说，把子图结果合并回父图
    return _runner  # 作为“节点”返回给父图

class PipelineState(TypedDict): # TypedDict用于给 dict 加上「键必须存在且类型固定」的静态约束。这相当于在类型层面声明：我们的「共享状态」一定包含这四个键。
    user_input:         str
    processed_input:    str
    route:              str # <─ new  by Router
    llm_json:           str
    execution_output:   Any # DataFrame / scalar / file path / str
# 调用 StateGraph(PipelineState) 时，LangGraph 会用这份类型信息来做静态检查：每个节点对 state 的读写都应该遵守这个 schema。这样能在开发期提前暴露字段拼写或类型错误。
# -------- ReAct 新增 --------
    scratchpad: str              # 记录 Thought / Action / Observation
    llm_output: str              # 本轮原始 LLM 回复
    action: dict | None          # 解析成功的 Action JSON
    final_answer: str | None
    step: int                    # 防死循环

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

# def flatten_cases(cases_route):
#     flat_route = []
#     for cond, dest in cases_route:
#         flat_route.extend([cond, dest])  # 依次压入 cond、dest
#     return flat_route

# ---------- build graph ---------------------------------------------------
def build_graph():
    sg = StateGraph(PipelineState)

    # root line
    sg.add_node("pre",      _pre) # "pre" 是节点名；_preprocess 为执行函数。运行时 Graph 把 state 交给该函数
    sg.add_node("router",   _route)

    sg.add_edge("pre", "router")

    # sub-graphs
    # tabular = build_tabular_subgraph()  # 返回编译好的 StateGraph
    # sg.add_subgraph("tabular", tabular) # add_subgraph 方法会把子图的所有节点和边添加到主图中（把它当成父图的“巨型节点”）。
    # sg.add_subgraph("tabular", build_tabular_react_subgraph())  # ReAct 子图
    # sg.add_subgraph("kg", build_kg_subgraph())
    # sg.add_subgraph("anomaly", build_anomaly_subgraph())
    # sg.add_subgraph("viz", build_viz_subgraph())
    # sg.add_subgraph() 已经过时了！！！LangGraph 现在推荐使用 add_node() 方法来添加子图！！！

    # ---------- sub-graphs (as nodes) ----------
    # ① Tabular-ReAct —— 父子字段一致，直接 passthrough
    sg.add_node(
        "tabular",
        passthrough(build_tabular_react_subgraph())
    )

    # # ② KG —— 只读 processed_input、写 execution_output
    # sg.add_node(
    #     "kg",
    #     wrap_with_mapping(
    #         build_kg_subgraph(),
    #         to_child=lambda ps: {"processed_input": ps["processed_input"]},
    #         # 返回值：新建一个仅含 processed_input 的字典。等价于：
    #         # def to_child(ps):
    #         #     return {"processed_input": ps["processed_input"]}
    #         from_child=lambda ps, cs: (
    #                 ps.update({"execution_output": cs["execution_output"]}) or ps
    #         # 把子图输出写回父-state 原地更新，该方法返回 None。由于前一半返回 None，None or ps 结果为 ps；这样整个表达式返回 更新后的父-state。
    #         # 也等价于：
    #         # def from_child(ps, cs):
    #         #     ps["execution_output"] = cs["execution_output"]
    #         #     return ps
    #         ),
    #     ),
    # )

    # ③ Anomaly
    sg.add_node("anomaly", passthrough(build_anomaly_subgraph()))

    # # ④ Viz
    # sg.add_node("viz", passthrough(build_viz_subgraph()))

    # router → conditional edges
    # sg.add_conditional_edges(
    #     "router",
    #     [
    #         (lambda s: s["route"] == "kg", "kg"),
    #         (lambda s: s["route"] == "anomaly", "anomaly"),
    #         (lambda s: s["route"] == "viz", "viz"),
    #         # 这里不要再用 list！
    #         (lambda _: True, "tabular"),
    #     ],
    # )

    # router_cases = [
    #     (lambda s: s["route"] == "kg", "kg"),
    #     (lambda s: s["route"] == "anomaly", "anomaly"),
    #     (lambda s: s["route"] == "viz", "viz"),
    # ]
    #
    # flat_router = [x for pair in router_cases for x in pair]
    #
    # sg.add_conditional_edges("router", *flat_router, "tabular")   # ← 兜底

    # # ---------- router 条件 ----------
    # def _router_switch(state: dict) -> str:
    #     match state["route"]:
    #         case "kg":      return "kg"
    #         case "anomaly": return "anomaly"
    #         case "viz":     return "viz"
    #     return "tabular"
    #
    # sg.add_conditional_edges("router", _router_switch)

    # —— 路由条件 ——
    sg.add_conditional_edges(
        "router",
        lambda s: "anomaly" if s["route"] == "anomaly" else "tabular"
    )

    # ---------- printer → END ----------
    def _print_final(state: dict):
        print("\n=== FINAL ANSWER ===")
        if state.get("final_answer"):
            print(state["final_answer"])
        else:  # 兜底：DataFrame 前 10 行
            df = state.get("execution_output")
            if hasattr(df, "head"):
                print(df.head(10).to_string(index=False))
            else:
                print(df)
        return state

    sg.add_node("printer", _print_final)
    sg.add_edge("tabular", "printer")
    sg.add_edge("anomaly", "printer")

    # 只画一次  router→tabular / kg / …  ，然后这些子图直接连 END
    sg.add_edge("tabular",  END)
    sg.add_edge("anomaly",  END)

    sg.set_entry_point("pre")
    return sg.compile()


