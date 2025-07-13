# rag_nodes_react/thought.py
# Thought-LLM 节点：拼 Prompt → 叫 LLM → 把回复写回 state。
# 取代旧 PromptingNode 在 Tabular 分支 中“一次全量生成计划”的功能。

from __future__ import annotations
from typing import Dict, Any
from string import Template
from RAG_tools import TOOL_REGISTRY
from openai import OpenAI
import os, pandas as pd, logging
import os, dotenv; dotenv.load_dotenv()

log = logging.getLogger("rag.thought")

# ---------- 环境 ----------
dotenv.load_dotenv()
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
# TOOL_SPEC = "\n".join(
#     f"### {name}\n{tool.description}" for name, tool in TOOL_REGISTRY.items()
# )
TOOL_SPEC = "\n".join(
    f"### {name}\n{tool.description}\nRequired keys: {', '.join(tool.signature)}"
    for name, tool in TOOL_REGISTRY.items()
)

_PROMPT_T = Template("""
You are an assistant that MUST translate a user's natural‑language request into a raw JSON command describing a sequence of data‑processing steps.
If the question restricts rows (e.g. “for Grinding jobs”), ALWAYS start with a select_rows action that applies that filter.

Decision hints:
– If the user asks for delay / duration between two time columns, first call add_derived_column OR directly call calculate_delay_avg / calculate_delay_avg_grouped instead of naïvely putting "colA - colB" into other tools.
– After aggregation (group_by_aggregate / group_top_n) do not re-aggregate the already-aggregated table unless the user explicitly asks so.
– If you still need to filter rows afterwards, DO NOT call *_avg tools; use add_derived_column or select_columns instead.


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
  
Additional formatting rule
--------------------------
• When you need to aggregate an *existing* column, ALWAYS use
  {
      "function": "group_by_aggregate",
      "args": {
          "agg"         : "<avg|sum|min|max|…>",
          "group_column": "<group key>",
          "column"      : "<target column>"
      }
  }
  DO NOT wrap the column inside "derived".

Example
~~~~~~~
User     : "Average Energy_Consumption by Operation_Type"
Assistant: {
             "actions":[
               {"function":"group_by_aggregate",
                "args":{"agg":"avg","group_column":"Operation_Type",
                        "column":"Energy_Consumption"}}
             ]
           }
""".lstrip())
# scratchpad: 是一个字符串，包含之前的 LLM 回复（观察），用于上下文。

# ---------- node ----------
def thought_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _PROMPT_T.substitute(
        tool_spec  = TOOL_SPEC,
        cols       = COLS,
        scratchpad = state.get("scratchpad", ""),
        user       = state["processed_input"],
    )

    log.debug("LLM prompt (first 400 chars):\n%s", prompt[:400])

    resp = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},   # ←★ 保证纯 JSON :contentReference[oaicite:3]{index=3}
        stop=["```"],                              # ←★ 防止 markdown 包裹 :contentReference[oaicite:4]{index=4}
    )

    state["llm_output"] = resp.choices[0].message.content.strip()
    log.debug("LLM raw output: %s", state["llm_output"])
    return state


