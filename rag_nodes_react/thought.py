# rag_nodes_react/thought.py
# Thought-LLM 节点：拼 Prompt → 叫 LLM → 把回复写回 state。
# 取代旧 PromptingNode 在 Tabular 分支 中“一次全量生成计划”的功能。

from typing import Dict, Any
from string import Template
from openai import OpenAI
from RAG_tools import TOOL_REGISTRY

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="NGC_API_KEY"
)

_PROMPT_T = Template("""
You are a ReAct agent working on a CSV table.
TOOLS = ${tools}

When you decide to act, output **only** a JSON, e.g.
{"function": "select_rows", "args": {...}}

If you have the final answer, output:
{"finish": "<answer>"}

Scratchpad:
${scratchpad}
User: ${user}
""".lstrip())

def thought_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sp   = state.get("scratchpad", "")  # 第一次循环 scratchpad 不存在 → 给空串；之后把旧内容取出来做上下文。
    user = state["processed_input"]     # 预处理阶段留下的英文/中译英文语句。

    prompt = _PROMPT_T.substitute(
        tools=", ".join(sorted(TOOL_REGISTRY.keys())),  # TOOL_REGISTRY.keys() → dict_keys 集合；sorted(...) → 排序成列表，保证顺序一致；用逗号+空格连成一条字符串
        scratchpad=sp,
        user=user
    )
    stream = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, top_p=0.7,
        max_tokens=1024, stream=True
    )
    rsp_text = ""
    for ch in stream:
        delta = ch.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            rsp_text += delta
    state["llm_output"] = rsp_text.strip()

    # 更新 scratchpad & 输出
    state["llm_output"] = stream               # 把模型回复存入状态，让 validator 用。
    state["scratchpad"] = sp + ("\n" if sp else "") + rsp_text   # 把新回复附加到历史，对 LLM 而言就是“Observation 已写入”。
    return state


