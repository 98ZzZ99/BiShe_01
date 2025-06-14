# RAG_subgraph_kg.py

from typing import Dict, Any
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
_LLM_KG_MODEL  = "meta/llama-3.1-70b-instruct"

def _stub_graph(msg: str):
    sg = StateGraph(dict)
    def _stub(state: Dict[str, Any]):          # 占位节点
        state["execution_output"] = msg
        return state
    sg.add_node("kg_stub", _stub)
    sg.set_entry_point("kg_stub")
    return sg.compile()

try:
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise ValueError("env-missing")

    from langchain_community.graphs import Neo4jGraph
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    from langchain_openai import ChatOpenAI

    _graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)  # 可能抛 DNS 或 Auth 错

    _qa = GraphCypherQAChain.from_llm(ChatOpenAI(model=_LLM_KG_MODEL, temperature=0),
                                      graph=_graph, top_k=10)

    def _kg(state: Dict[str, Any]) -> Dict[str, Any]:
        state["execution_output"] = _qa.run(state["processed_input"])
        return state

    def build_kg_subgraph():
        sg = StateGraph(dict)
        sg.add_node("kg", _kg)
        sg.set_entry_point("kg")
        return sg.compile()

except Exception as err:
    msg = f"[KG 初始化失败: {err.__class__.__name__}]"
    build_kg_subgraph = lambda: _stub_graph(msg)


