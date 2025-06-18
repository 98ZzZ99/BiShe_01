# RAG_node_2_execution.py

# 可以删除！！！！！

"""ExecutionNode — 读取 JSON actions → 调用工具 → 返回最终结果。"""
import json, pandas as pd
from RAG_tools import TOOL_REGISTRY, reset_state

class ExecutionNode:
    def __init__(self) -> None: # 构造器 __init__ 是 Python 类的实例化钩子；当你写 _exec = ExecutionNode() 时会自动触发。
        print("[LOG] ExecutionNode initialized.")

    def _call_tool(self, fname: str, args: dict):
        if fname not in TOOL_REGISTRY:
            raise ValueError(f"Unknown function: {fname}")
        tool = TOOL_REGISTRY[fname]
        return tool.invoke(json.dumps(args) if args else "")

    def run(self, llm_json_str: str):
        print("[LOG] ExecutionNode running …")
        reset_state()
        plan = json.loads(llm_json_str) # JSON → Python dict。json.loads：标准库函数，把 JSON 字符串反序列化为 Python 对象。
        actions = plan.get("actions", [])   # 取 "actions" 列表，若缺省则给空。dict.get是安全地取键，若不存在返回默认值（这里是空列表）。
        last_result = None  # 初始占位：方便循环结束后还能访问到「最后一步」结果。
        for step in actions:    # actions 是列表，元素来自刚解析的 JSON；每个元素是一步加工计划。
            fname = step["function"]    # 要调用的工具名
            args  = step.get("args", {})    # 传参字典
            print(f"[LOG] → {fname}  args={args}")
            last_result = self._call_tool(fname, args)
        # 打印 DataFrame/文件预览
        if isinstance(last_result, pd.DataFrame):   # isinstance()：检测对象是否属于某个类型或其子类。pd.DataFrame是 Pandas 核心二维表对象。
            print("[LOG] Final DataFrame (head):")
            print(last_result.head(10).to_string(index=False))
        else:
            print("[LOG] Final result:", last_result)
        return last_result


