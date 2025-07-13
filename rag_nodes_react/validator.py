# rag_nodes_react/validator.py
from __future__ import annotations
import json, logging
from json import JSONDecodeError
from typing import Dict, Any
from pydantic import ValidationError
from json_repair import repair_json      # :contentReference[oaicite:5]{index=5}
from .models import Action, Finish

log = logging.getLogger("rag.validator")

FUNC_ALIAS = {
    # 针对 group_by_aggregate
    "group_by_aggregate": {
        "value_column": "target_column",
        "return_direct": "target_column",
        "agg_column":    "target_column",
    },
    # 针对 group_top_n
    "group_top_n": {
        "sort_column": "column",
    },
    # covariance / correlation 两列写在 columns 数组
    "calculate_covariance": {"columns->": ("x", "y")},
    "calculate_correlation": {"columns->": ("x", "y")},
    # “Job_Type” 或 “job type” 被用户/LLM写出来时会自动替换成真实列 Operation_Type
    "job_type": "Operation_Type",
}

MAX_RETRY = 3

def _alias(args: dict) -> dict:
    return {FUNC_ALIAS.get(k, k): v for k, v in args.items()}

def _normalize(act: dict) -> dict:
    """Rename LLM-supplied aliases so the downstream tool sees a stable schema"""
    ALIAS = {
        # 单列
        "column": "target_column",        # for select_rows, etc.
        "value_column": "target_column",
        "return_column": "target_column",
        "return_direct": "target_column",

        # 双列 (协方差/相关)
        "x": "target_column",
        "y": "other_column",
        "columns": "pair",                # ["A","B"] → x=A, y=B

        # top-n
        "sort_column": "column",
    }

    a = act.get("args", {})

    # 1) pair expansion
    if a.get("pair") and len(a["pair"]) == 2:
        a["x"], a["y"] = a.pop("pair")

    # 2) rename keys in-place
    for k_old, k_new in list(a.items()):
        if k_old in ALIAS:
            a[ALIAS[k_old]] = a.pop(k_old)

    # 3) percentile always float
    if "q" in a:  # 90 → 90.0
        a["q"] = float(a["q"])

    act["args"] = a
    return act

def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ---------- 队列未清空，直接回执行 ----------
    if state.get("route") == "finish":
        return state

    if state.get("action_queue"):          # 还有待执行步骤
        state["route"] = "execute"
        return state

    raw = state.pop("llm_output", "")        # ➊ 取出后立即 pop，避免下一轮重复解析
    step = state.get("step", 0) + 1
    state["step"] = step


    if step > MAX_RETRY:
        state.update(route="finish", final_answer="[Error] too many retries")
        log.warning("Exceeded max retry, giving up.")
        return state

    log.debug("Validator step %s | raw (200 chars): %s", step, raw[:200])

    # ---------- 解析 JSON ----------

    try:
        l, r = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[l:r + 1])
    except (JSONDecodeError, Exception) as e:
        state.update(route="error", observation=f"[JSON-Error] {e}")
        log.error("JSON decode failed: %s", e)
        return state

    # ---------- finish 分支 ----------

    # if "finish" in data:
    #
    #     try:
    #         fin = Finish.model_validate(data)
    #         state.update(route="finish", final_answer=fin.finish)
    #         log.debug("Finish branch with answer: %s", fin.finish)
    #         return state
    #     except ValidationError as e:
    #         state.update(route="error", observation=f"[Finish-Validation] {e}")
    #         log.error("Finish validation error: %s", e)
    #         return state
    #
    # # ---------- actions 分支 ----------
    # actions = data.get("actions") or [data]
    # try:
    #     state["action_queue"] = [
    #         Action.model_validate(_normalize(a)).model_dump() for a in actions
    #     ]
    #     state["route"] = "execute"
    #     log.debug("Validated %d action(s) -> route execute", len(actions))
    #     return state
    # except ValidationError as e:
    #     state.update(route="error", observation=f"[Action-Validation] {e}")
    #     log.error("Action validation error: %s", e)
    #     return state


    # -------- 生成 action_queue --------
    try:
        if "actions" in data:                                    # 多步
            acts = [_normalize(a) for a in data["actions"]]
        elif "finish" in data:                                   # 一步回答
            state.update(route="finish", final_answer=data["finish"])
            return state
        else:                                                    # 单 action
            acts = [_normalize(data)]

        state["action_queue"] = [Action.model_validate(a).model_dump()
                                  for a in acts]
        state["route"] = "execute"               # ❷ 永远只发往 execute
        return state
    except ValidationError as e:
        state.update(route="error", observation=f"[Action-Validation] {e}")
        return state

