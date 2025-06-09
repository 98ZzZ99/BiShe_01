# RAG_subgraph_tabular.py

"""
TabularSubGraph = PrePrompt → Prompt → Execute
Re-use your existing PromptingNode / ExecutionNode classes.
"""
from typing import Dict, Any
from langgraph.graph import StateGraph

from RAG_node_1_prompting import PromptingNode
from RAG_node_2_execution import ExecutionNode

_Prompt = PromptingNode()
_Exec   = ExecutionNode()

# 调用现有 PromptingNode，让 第二层 LLM 生成 JSON
def _prompt_stage(state:Dict[str,Any])->Dict[str,Any]:
    state["llm_json"] = _Prompt.run(state["processed_input"])
    return state

# 调用 ExecutionNode，执行 JSON 里的工具
def _execute_stage(state:Dict[str,Any])->Dict[str,Any]:
    state["execution_output"] = _Exec.run(state["llm_json"])
    return state

def build_tabular_subgraph():
    sg = StateGraph(dict)        # 子图内部单独用 dict
    sg.add_node("prompt",  _prompt_stage)
    sg.add_node("execute", _execute_stage)
    sg.add_edge("prompt", "execute")
    sg.set_entry_point("prompt")
    return sg.compile()

def placeholder(label:str):
    def _run(state):
        state["execution_output"] = f"[{label} branch not implemented]"
        return state
    return _run
# 这个 placeholder 函数是为了在主图里占位，避免报错。实际使用时会被具体的子图函数替换。


