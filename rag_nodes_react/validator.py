# rag_nodes_react/validator.py
# 对 LLM 输出做 json.loads + Pydantic 校验，决定下一条边走 execute 还是 error 还是 finish。
# 与 thought.py 和 execute.py 形成一个 ReAct 循环。

from typing import Dict, Any
from json import JSONDecodeError, loads
from pydantic import ValidationError
from .models import Action, Finish

def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    txt = state["llm_output"]
    try:
        obj = loads(txt)
        # ① 是否 “finish”
        # 校验成功且含 "finish" → 把答案写 state["final_answer"]，state["route"]="finish"
        if "finish" in obj:
            state["final_answer"] = Finish.model_validate(obj).finish
            state["route"] = "finish"
            return state
        # ② 尝试校验 Action
        # 校验成功且含 Action → state["action"]=…, route="execute"
        action = Action.model_validate(obj)
        state["action"] = action.model_dump()
        state["route"] = "execute"
    except (JSONDecodeError, ValidationError) as err:
        # ③ JSONDecodeError 或 ValidationError → state["route"]="error"写下出错原因
        state["observation"] = f"[JSON-Error] {err}"
        state["route"] = "error"
    return state


