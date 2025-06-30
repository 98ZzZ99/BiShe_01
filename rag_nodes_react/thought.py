# rag_nodes_react/thought.py
# Thought-LLM 节点：拼 Prompt → 叫 LLM → 把回复写回 state。
# 取代旧 PromptingNode 在 Tabular 分支 中“一次全量生成计划”的功能。

from typing import Dict, Any
from string import Template
from RAG_tools import TOOL_REGISTRY
# from dotenv import load_dotenv; load_dotenv()
# from dotenv import load_dotenv; load_dotenv()
import os
from openai import OpenAI
import pandas as pd
import os, dotenv; dotenv.load_dotenv()

# ---------- 环境 ----------
dotenv.load_dotenv()
print("[DEBUG] key =", os.getenv("NGC_API_KEY"))

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NGC_API_KEY"),
)

# ---------- 列名 & 工具规范 ----------
# 读取数据文件的列名，供工具函数使用
# 在 预处理 阶段如检测到用户显式提供了 some_file.csv，load_data() 会用该文件；否则回落默认 data/hybrid_manufacturing_categorical.csv。
# OLS 只用于给 LLM 提供「列名全集」，生产环境可在读取预处理结果后 重新 计算列名并写回 prompt；为演示简化成固定文件，无功能冲突。
COLS = ", ".join(pd.read_csv("data/hybrid_manufacturing_categorical.csv", nrows=0).columns)

# # TOOL_REGISTRY 是 RAG_tools.py 中的全局变量，包含所有工具函数的注册表。
# # args 是传给工具函数的参数 dict
# TOOL_SPEC = """
# ### filter_date_range
# args: key–value pairs required by that function
# """.strip()
# # .strip() 去掉首尾空白行，避免多余空行干扰。

# ---------- 动态拼接全部工具描述 ----------
# \n 是换行符；.join() 把多行字符串串起来；f"### …" 是 Markdown 标题格式。
# 这样 prompt 中就带「函数名 + 简短功能说明」，LLM 不会再只看到固定几行。
TOOL_SPEC = "\n".join(
    f"### {name}\n{tool.description}" for name, tool in TOOL_REGISTRY.items()
)

_PROMPT_T = Template("""
You are an assistant that MUST translate a user's natural‑language request into a raw JSON command describing a sequence of data‑processing steps.
If the question restricts rows (e.g. “for Grinding jobs”), ALWAYS start with a select_rows action that applies that filter.

$tool_spec

Table Columns: $cols

JSON schema you MUST follow **for each step**:
{"function": "<one of the tool_name>", "args": { ... } }

Example:
（It does not mean that you must return the same command as this example, but just similar ideas）
        User input:
        "I need all jobs whose Optimization_Category is Low Efficiency AND
        Processing_Time ≤ 50, then sort them by Machine_Availability descending."

        Expected JSON output (RAW, no markdown):
        {
          "actions": [
            { "function": "select_rows",
              "args": { "column": "Optimization_Category", "condition": "== 'Low Efficiency'" }
            },
            { "function": "select_rows",
              "args": { "column": "Processing_Time", "condition": "<= 50" }
            },
            { "function": "sort_rows",
              "args": { "column": "Machine_Availability", "order": "desc" }
            }
          ]
        }

If you have the final answer, output:
{"finish": "<answer>"}

Scratchpad:
${scratchpad}
User: ${user}

Return either  
  • exactly ONE such object, **not wrapped in an "actions" list**  
  • or {"finish":"<answer>"} if you are done.
""".lstrip())
# scratchpad: 是一个字符串，包含之前的 LLM 回复（观察），用于上下文。

# ---------- node ----------
def thought_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # sp   = state.get("scratchpad", "")  # 第一次循环 scratchpad 不存在 → 给空串；之后把旧内容取出来做上下文。
    # user = state["processed_input"]     # 预处理阶段留下的英文/中译英文语句。

    # .substitute()的方法，用实参替换模板里的 $占位符。
    prompt = _PROMPT_T.substitute(
        tool_spec   = TOOL_SPEC,
        cols        = COLS,
        scratchpad  = state.get("scratchpad", ""),
        user        = state["processed_input"]
    )

    resp = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
        stream=False,
    )

    print("[LLM OUTPUT]", resp.choices[0].message.content[:400])
    state["llm_output"] = resp.choices[0].message.content.strip()
    print("[PROMPT]", prompt[:400])
    return state

# {_PROMPT_T.substitute(tools=", ".join(TOOL_REGISTRY), scratchpad=sp, user=user)}
# """
#     # stream = client.chat.completions.create(
#     #     model="meta/llama-3.1-70b-instruct",
#     #     messages=[{"role": "user", "content": prompt}],
#     #     temperature=0.2, top_p=0.7,
#     #     max_tokens=1024, stream=True
#     # )
#     # rsp_text = ""
#     # for ch in stream:
#     #     delta = ch.choices[0].delta.content
#     #     if delta:
#     #         print(delta, end="", flush=True)
#     #         rsp_text += delta
#     # state["llm_output"] = rsp_text.strip()
#
#
#     # 更新 scratchpad & 输出
#     # state["llm_output"] = stream               # 把模型回复存入状态，让 validator 用。
#     # state["scratchpad"] = sp + ("\n" if sp else "") + rsp_text   # 把新回复附加到历史，对 LLM 而言就是“Observation 已写入”。
#
#     resp = client.chat.completions.create(
#         model="meta/llama-3.1-8b-instruct",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=1024,
#         stream=False,                         # ←★★ 关掉流式
#     )
#     content = resp.choices[0].message.content
#     state["llm_output"] = content            # downstream 看到的就是纯 str
#
#     return state


