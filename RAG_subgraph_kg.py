# RAG_subgraph_kg.py

from typing import Dict, Any
from langchain.graphs import Neo4jGraph
from langchain.chains.graph_qa.cypher import GraphCypherQAChain  # :contentReference[oaicite:9]{index=9}
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

import os

NEO4J_URI      = os.getenv("neo4j+s://ca96d1d6.databases.neo4j.io")         # 从环境变量读取连接串、用户名、密码；类似读取 API-KEY。
NEO4J_USER     = os.getenv("neo4j")
NEO4J_PASSWORD = os.getenv("yXdQU5l5Ut1GB6zo_ISjcbp73CGz-KbSE-pA1pRN-84")
_LLM_KG_MODEL  = "meta/llama-3.1-70b-instruct"

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD) # 创建 LangChain 数据库驱动包装器，内部会通过 Bolt 协议连线。

qa_chain = GraphCypherQAChain.from_llm(     # LangChain 现成链：①让 LLM 把自然语句翻成 Cypher；②执行；③把结果再转成自然语言。
    ChatOpenAI(model=_LLM_KG_MODEL, temperature=0),
    graph=graph,
    top_k=10,
)

# _kg_stage 是子图里的唯一节点：取 processed_input → qa_chain.run() → 把答案写回 state["execution_output"]。
def _kg_stage(state:Dict[str,Any])->Dict[str,Any]:
    question = state["processed_input"]
    answer   = qa_chain.run(question)
    state["execution_output"] = answer
    return state

def build_kg_subgraph():
    sg = StateGraph(dict)
    sg.add_node("kg", _kg_stage)
    sg.set_entry_point("kg")
    return sg.compile()


