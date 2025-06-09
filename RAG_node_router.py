# RAG_node_router.py

"""
RouterNode — decide which domain sub-graph should handle the request.
Modes:
  rule  : simple keyword → label mapping
  llm   : call small LLM to classify intent
"""
from __future__ import annotations
import os, re, json
from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()   # python-dotenv 库会把同目录或父目录的 .env 文件读取出来，并写入 os.environ
ROUTER_MODE: Literal["rule","llm"] = os.getenv("ROUTER_MODE","rule").lower()  # typing.Literal 指定变量只能取这两个常量之一。读取环境变量 ROUTER_MODE，若不存在则默认 "rule"。把结果转小写，容忍用户写 RULE、Rule 等。
NGC_API_KEY = os.getenv("NGC_API_KEY")        # 若用 llm 模式

# --- keyword sets ---------------------------------------------------------
_KEYWORDS = {   # 字典
    "kg"      : {"graph","cypher","relationship","path","neo4j"},
    "anomaly" : {"anomaly","outlier","异常","离群","grubbs","k-nn"},
    "viz"     : {"plot","chart","visualise","可视化","画图","histogram"},
}

class RouterNode:
    """Return dict(route=<tabular|kg|anomaly|viz|unknown>)"""

    def __init__(self) -> None:
        print(f"[LOG] RouterNode initialized (mode={ROUTER_MODE}).")
        if ROUTER_MODE == "llm":
            if NGC_API_KEY is None:
                raise EnvironmentError("NGC_API_KEY not set for llm router")
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=NGC_API_KEY,
                timeout=30,
            )

    # -------------- rule-based -------------------------------------------
    def _route_by_rule(self, text:str)->str:
        lower = text.lower()                    # 全部转小写，方便不区分大小写匹配
        for label, kws in _KEYWORDS.items():    # kws 是每个标签对应的 set
            if any(kw in lower for kw in kws):  # kw 是 for-comprehension 中的迭代变量，代表关键词集合里的某个关键字字符串；kws 是该集合本身。
                return label
        return "tabular"       # 默认走表格分支

    # -------------- llm-based --------------------------------------------
    _SYSTEM_MSG = (
        "You are a routing assistant. "
        "Return ONLY one token from {tabular, kg, anomaly, viz}. "
        "Decision rules:\n"
        "- If the user asks about relationships, paths or Cypher ⇒ kg\n"
        "- If the user asks for anomaly/outlier detection ⇒ anomaly\n"
        "- If the user asks to plot/visualise ⇒ viz\n"
        "- Otherwise ⇒ tabular"
    )

    def _route_by_llm(self, text:str)->str:
        resp = self.client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role":"system","content":self._SYSTEM_MSG},   # system 用来注入指令
                {"role":"user"  ,"content":text.strip()}        # user 放实际输入
            ],
            temperature=0.0,
            max_tokens=1,
        )
        label = resp.choices[0].message.content.strip().lower() # resp：OpenAI 返回的响应对象。.choices[0]：第一条候选答案。.message.content：该答案的文本。.strip().lower()：去掉首尾空白再转小写。
        if label not in {"tabular","kg","anomaly","viz"}:
            label = "tabular"
        return label

    # -------------- public API -------------------------------------------
    def run(self, processed_input:str)->dict:
        if ROUTER_MODE == "llm":
            label = self._route_by_llm(processed_input)     # 快、可解释、无需 API；关键词漏判、需手工维护
        else:
            label = self._route_by_rule(processed_input)    # 语义鲁棒、易扩展； 需 API/计费、慢、可能出错
        print(f"[LOG] Router decision → {label}")
        return {"route": label}


