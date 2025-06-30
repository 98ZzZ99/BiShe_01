# rag_nodes_react/validator.py
# 对 LLM 输出做 json.loads + Pydantic 校验，决定下一条边走 execute 还是 error 还是 finish。
# 与 thought.py 和 execute.py 形成一个 ReAct 循环。

from typing import Dict, Any
from json import JSONDecodeError, loads
from pydantic import ValidationError
from .models import Action, Finish
from traceback import format_exc    # traceback 用于 追踪异常栈信息。format_exc() 可以 返回 最近一次异常的完整 stack-trace 字符串。
import re, json

def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:

    # 检查 state["route"] 是否是 "finish"
    if state.get("route") == "finish":
        return state  # 已经结束，不要再解析

    # 防止死循环
    MAX_STEP = 3
    state["step"] = state.get("step", 0) + 1
    if state["step"] > MAX_STEP:
        state["final_answer"] = "[Error] too many retries"
        state["route"] = "finish"
        return state

    print("[DEBUG] validator got:",
          (state.get("llm_output") or "")[:300],  # 只截前 300 字符，避免刷屏
          flush=True)

    txt = state["llm_output"]

    try:
        # 1) 清理前缀
        raw = state["llm_output"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
        raw = raw.replace("\u201c", "\"").replace("\u201d", "\"")

        # 2) 截取最大花括号块   —— 仅 3 行
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise JSONDecodeError("no JSON found", raw, 0)
        json_str = raw[start:end]

        obj = json.loads(json_str)

        # ① 是否 “finish”
        # 校验成功且含 "finish" → 把答案写 state["final_answer"]，state["route"]="finish"
        # if "finish" in obj:
        #     state["final_answer"] = Finish.model_validate(obj).finish
        #     state["route"] = "finish"
        #     return state
        #
        # # ② 尝试校验 Action
        # # 校验成功且含 Action → state["action"]=…, route="execute"
        # action = Action.model_validate(obj)
        # state["action"] = action.model_dump()
        # state["route"] = "execute"
        if "actions" in obj:
            if not obj["actions"]:
                raise ValidationError("actions list empty")
            obj = obj["actions"][0]  # 只取第一条
        action = Action.model_validate(obj)  # 再走 Pydantic
        state["action"] = action.model_dump()
        state["route"] = "execute"

    except (JSONDecodeError, ValidationError) as err:
        print("[VALIDATOR ERROR]", err)
        print("[PAYLOAD]", raw[:500])  # 打印前几百字符
        print("[STACK]", format_exc())  # 堆栈
        # ③ JSONDecodeError 或 ValidationError → state["route"]="error"写下出错原因
        state["observation"] = f"[JSON-Error] {err}"
        state["route"] = "error"

    return state


