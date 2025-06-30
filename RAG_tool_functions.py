# RAG_tool_functions.py
# 说明：此文件包含“选择”、“排序”、“平均”、“中位数”、“众数”等工具函数
#       以及对 CSV 数据的加载和处理逻辑
"""
全局 DataFrame → DataFrame
select_rows（支持 AND/OR）、sort_rows、top_n、group_top_n、filter_date_range、add_derived_column、rolling_average

统计标量
calculate_average、median、mode、sum、min、max、std、variance、percentile、correlation、covariance

专用
calculate_failure_rate、calculate_delay_avg
"""

import os, re
import pandas as pd
import numpy as np

# ---------- 基础 ----------
TIME_COLS = ["Scheduled_Start", "Scheduled_End", "Actual_Start", "Actual_End"]
CSV_FILE = os.path.join("data", "hybrid_manufacturing_categorical.csv")

# ----------- 通用小工具 -----------
def _df(cur):  # 始终返回一个 DataFrame；避免布尔歧义
    return cur if cur is not None else load_data()

#def load_data():
#     if not os.path.exists(CSV_FILE):
#         raise FileNotFoundError(CSV_FILE)
#     return pd.read_csv(CSV_FILE, parse_dates=TIME_COLS, dayfirst=False)

def load_data(path: str | None = None):
    """
    如果外部传入 path ⇒ 读取该文件；否则读默认 CSV_FILE。
    这样旧代码 (load_data()) 和新代码 (load_data(some_path)) 都能工作。
    """
    path = path or CSV_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=TIME_COLS, dayfirst=False)

def _num(s):
    """安全转数值"""
    return pd.to_numeric(s, errors="coerce")


# ---------- 1) 行级函数 (DF→DF) ----------
def select_rows(cur, args):
    """
    单列筛选 + 可选 AND/OR 第二条件
      args = { "column": "Processing_Time",
               "condition": ">= 50 AND <= 120" }
    逻辑顺序：
        先处理 AND/OR 递归；
        再拆 <op> <value>；
        紧接着立刻检查是否是 “HH:MM” 纯时间；
        其余情况落到原有日期 / 数值流程。
    """
    if cur is None:
        cur = load_data()

    col  = args["column"]
    cond = args["condition"]

    # ---------- 1.1) 递归处理 AND / OR ----------
    if m := re.search(r"\s+(AND|OR)\s+", cond, flags=re.I):
        left, op_logic, right = re.split(r"\s+(AND|OR)\s+", cond, 1, flags=re.I)
        df1 = select_rows(cur, {"column": col, "condition": left})
        df2 = select_rows(cur, {"column": col, "condition": right})
        return (pd.merge(df1, df2) if op_logic.upper() == "AND"
                else pd.concat([df1, df2]).drop_duplicates())

    # ---------- 1.2) 解析 <op> <value> ----------
    m = re.match(r"^(==|!=|<=|>=|<|>)\s*(.+)$", cond.strip())
    if not m:
        raise ValueError("Bad condition syntax")
    op, val_raw = m.groups()
    val_raw = val_raw.strip(' "\'')

    # ---------- 1.3) 仅“HH:MM”时间分量比较 ----------
    # 必须放在任何 return 之前！
    if re.fullmatch(r"\d{1,2}:\d{2}", val_raw):
        t_val = pd.to_datetime(val_raw, format="%H:%M").time()
        dt_series = pd.to_datetime(cur[col], errors="coerce").dt.time
        ops = {"==": "eq", "!=": "ne", "<": "lt",
               "<=": "le", ">": "gt", ">=": "ge"}
        mask = getattr(dt_series, ops[op])(t_val)
        return cur[mask]

    # ---------- 1.4) 常规日期 / 数值 / 字符串 ----------
    if col in TIME_COLS:
        # 仅有日期时默认补 00:00
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", val_raw):
            val_raw += " 00:00"
        val = pd.to_datetime(val_raw, errors="coerce")
    else:
        try:
            val = float(val_raw)
        except ValueError:
            val = val_raw

    ops = {"==": "eq", "!=": "ne", "<": "lt",
           "<=": "le", ">": "gt", ">=": "ge"}
    mask = getattr(cur[col], ops[op])(val)
    return cur[mask]


def sort_rows(cur, args):
    if cur is None: cur = load_data()
    return cur.sort_values(args["column"], ascending=args.get("order","asc")=="asc")

def top_n(cur, args):
    if cur is None: cur = load_data()
    n   = int(args.get("n",5))
    asc = args.get("order","desc")!="desc"
    return cur.sort_values(args["column"], ascending=asc).head(n)

def group_top_n(cur, args):
    """
    每组取前 N
      args = {"group_column":"Machine_ID",
              "sort_column":"Processing_Time",
              "order":"desc","n":2,"keep_all":True}
    """
    if cur is None: cur = load_data()
    g,s,n = args["group_column"], args["sort_column"], int(args.get("n",1))
    asc   = args.get("order","desc")!="desc"
    out = (cur.sort_values(s, ascending=asc)
              .groupby(g, as_index=False).head(n))
    keep = args.get("keep_all", True)

    if keep:
        return out
    # 即使 keep_all = false 也保留核心标识列，避免后续 KeyError
    id_cols = [c for c in ("Job_ID", "Machine_ID") if c in out.columns]
    return out[id_cols + [g, s]]

def filter_date_between_start_end(cur, args):
    """
    快捷时间窗口
      args = {"column":"Actual_Start",
              "start":"2023-03-18 10:00",
              "end":"2023-03-18 12:00",
              "inclusive":"both"}
    """
    if cur is None: cur = load_data()
    col = args["column"]
    s   = pd.to_datetime(args.get("start")) if args.get("start") else None
    e   = pd.to_datetime(args.get("end"))   if args.get("end")   else None
    inc = args.get("inclusive","both")
    ser = pd.to_datetime(cur[col], errors="coerce")
    mask = pd.Series(True, index=cur.index)
    if s is not None: mask &= ser.ge(s) if inc in ("both","left") else ser.gt(s)
    if e is not None: mask &= ser.le(e) if inc in ("both","right") else ser.lt(e)
    return cur[mask]

def add_derived_column(cur, args):
    """
    新增表达式列
      args = {"name":"EC_per_PT",
              "formula":"Energy_Consumption / Processing_Time"}
    问题1：未设置时报错
    解决方案：
        在执行替换时，若 self.last_scalar is None 则直接照字面保留，不报错
    问题2：派生列 Scheduled_Start - Actual_Start 抛 numexpr 错
    解决方案：
        执行 cur.eval 前，若检测到日期列差分，改为 pd.to_datetime(col1) - pd.to_datetime(col2) 并转 .dt.total_seconds()
    """
    formula = args["formula"].replace("{last_scalar}",
                                      str(last_scalar) if (
                                          last_scalar := globals().get("_LAST_SCALAR")) else "{last_scalar}")

    if " - " in formula and any(c in formula for c in TIME_COLS):
        lhs, rhs = [s.strip() for s in formula.split("-", 1)]
        delta = pd.to_datetime(cur[lhs]) - pd.to_datetime(cur[rhs])
        cur[args["name"]] = delta.dt.total_seconds()
    else:
        cur[args["name"]] = cur.eval(formula)
    return cur

def rolling_average(cur, args):
    """
    滚动平均    之前的 rolling_average(无分组) 返回 DF 中无原列，而 LLM 仍按原列排序。 修改后让函数始终 return DataFrame 并保留原列
    """
    if cur is None: cur = load_data()
    w   = int(args.get("window",3))
    col = args["column"]
    g   = args.get("group_by")
    if g:                                      # 分组滚动
        df = (cur.sort_values(g)    # 把多级索引行号改名为 row_idx，信息友好且不影响后续运算。
                 .groupby(g, group_keys=False)[col]
                 .rolling(w, min_periods=1).mean()
                 .reset_index())
        df.rename(columns={col:f"rolling_avg_{col}"}, inplace=True)
        return df
    else:                                     # 全局滚动
        ser = _num(cur[col]).rolling(w, min_periods=1).mean()
        return cur.assign(**{f"rolling_avg_{col}": ser})

# ---------- 2) group_by_aggregate ----------
def group_by_aggregate(cur, args):
    """
    支持 agg：avg/sum/min/max/count/std/var
    支持派生列 {"derived":{...}}
    elif agg == "percentile":
        q = float(args.get("q", args.get("percentile", 90)))
        grouped = target.groupby(cur[g]).quantile(q/100)
    """
    df = _df(cur); g = args["group_column"]
    keep = args.get("keep_all", False)
    agg = args.get("agg", "avg").lower()

    # 扩展 agg_map + percentile/cov/corr，增加了功能

    if agg == "percentile":
        q = float(args.get("q", args.get("percentile", 90)))
        res = (_num(cur[args["target_column"]])
               .groupby(cur[g])
               .quantile(q / 100)
               .reset_index(name=f"p{int(q)}_{args['target_column']}"))
        return cur.merge(res, on=g, how="left") if keep else res

    if agg in {"cov", "corr"}:  # 支持 cov / corr
        other = args.get("other_column")
        if other is None: raise ValueError("other_column required for cov/corr")
        grp = cur.groupby(g)
        func = "cov" if agg == "cov" else "corr"
        res = grp.apply(lambda df: _num(df[args["target_column"]]).__getattribute__(func)(_num(df[other])))
        res = res.reset_index(name=f"{agg}_{args['target_column']}_{other}")
        return cur.merge(res, on=g, how="left") if keep else res

    # ------- 目标列或派生 --------
    if "derived" in args:   # 此段是为了正确处理 derived["type"]（但在后续开发中进行了一定修改），把 timedelta 放到第一分支，彻底消除 “Unsupported derived type”
        d = args["derived"]
        if d["type"] == "timedelta":
            delta = (pd.to_datetime(df[d["end_col"]]) -
                     pd.to_datetime(df[d["start_col"]])).dt.total_seconds()
            if d.get("unit") == "minutes":
                delta /= 60
            elif d.get("unit") == "hours":
                delta /= 3600
            target = delta
            colname = d.get("name", "derived")
        else:
            raise ValueError("Unsupported derived type")
    else:
        colname = args["target_column"]
        target = _num(df[colname])

    # ------- 聚合 ---------  之前的版本 1.不支持 cov/percentile；2.keep_all=True 且无 derived 时 target 未定义；3.只有 timedelta 派生类型
    if agg in {"cov", "corr"}:  # 双列聚合
        other = _num(df[args["other_column"]])
        func = pd.Series.cov if agg == "cov" else pd.Series.corr
        res = (target.groupby(df[g])
               .apply(lambda s: func(s, other.loc[s.index]))
               .reset_index(name=f"{agg}_{colname}"))
    elif agg == "percentile":
        q = float(args.get("percentile", args.get("q", 90))) / 100
        res = (target.groupby(df[g]).quantile(q)
               .reset_index(name=f"p{int(q * 100)}_{colname}"))
    else:
        agg_map = {"avg": "mean", "mean": "mean", "sum": "sum", "min": "min", "max": "max",
                   "count": "count", "std": "std", "var": "var"}
        if agg not in agg_map: raise ValueError("Bad agg")
        res = getattr(target.groupby(df[g]), agg_map[agg])() \
            .reset_index(name=f"{agg}_{colname}")

    return df.merge(res, on=g) if keep else res


# ---------- 3) 标量 ----------
def calculate_average(cur,args):
    df = _df(cur); return _num(df[args["column"]]).mean()
def calculate_median(cur,args):
    df = _df(cur); return _num(df[args["column"]]).median()
def calculate_mode(cur,args):
    ser = (cur if cur is not None else load_data())[args["column"]].mode()
    return ser.iloc[0] if not ser.empty else None
def calculate_sum(cur,args):
    df=_df(cur); return _num(df[args["column"]]).sum()
def calculate_min(cur,args):
    df=_df(cur); return _num(df[args["column"]]).min()
def calculate_max(cur,args):
    df=_df(cur); return _num(df[args["column"]]).max()
def calculate_std(cur,args):
    df=_df(cur); return _num(df[args["column"]]).std()
def calculate_variance(cur,args):
    df=_df(cur); return _num(df[args["column"]]).var()
def calculate_percentile(cur,args):
    df=_df(cur); q=float(args.get("percentile", args.get("q",90)))
    g = args.get("group_by") or args.get("group_column")
    if g:
        return (df.groupby(g)[args["column"]]
                  .quantile(q/100)
                  .reset_index(name=f"p{int(q)}_{args['column']}"))
    return _num(df[args["column"]]).quantile(q/100)
def calculate_correlation(cur,args):
    x = args.get("x") or args.get("column1")
    y = args.get("y") or args.get("column2")
    df=_df(cur)
    return _num(df[x]).corr(_num(df[y]))
def calculate_covariance(cur,args):
    x = args.get("x") or args.get("column1")
    y = args.get("y") or args.get("column2")
    df=_df(cur)
    return _num(df[x]).cov(_num(df[y]))

# ---------- 新增 count_rows ----------
def count_rows(cur,args=None):
    df=_df(cur)
    return int(len(df))

# ---------- 4) 业务专用 ----------
def calculate_delay_avg(cur,args=None):
    args = args or {}
    unit = args.get("unit", "seconds")     # seconds|minutes|hours
    use_abs = bool(args.get("abs", False)) # 取绝对值？

    df = _df(cur)
    completed = df[df["Job_Status"] == "Completed"]
    delta = (pd.to_datetime(completed["Actual_End"]) -
             pd.to_datetime(completed["Scheduled_End"])).dt.total_seconds()

    if use_abs:
        delta = delta.abs()

    if unit == "minutes":
        delta /= 60
    elif unit == "hours":
        delta /= 3600

    return delta.mean()

def calculate_failure_rate(cur,args):
    """
    Failed / total per group_column
      args={"group_column":"Machine_ID"}
    """
    df = cur if cur is not None else load_data()
    g = args["group_column"]
    failed = df[df["Job_Status"]=="Failed"].groupby(g).size()
    total  = df.groupby(g).size()
    return (failed/total).fillna(0).reset_index(name="failure_rate")

# --------- 新增：延迟平均（分组）-----------
def calculate_delay_avg_grouped(cur, args):
    """
    • Average (Actual_End − Scheduled_End) per group.
    • args = { "group_column": <str>, "unit": "seconds|minutes|hours" }
    """
    df = cur if cur is not None else load_data()
    gcol = args["group_column"]
    unit = args.get("unit","seconds")
    delta = (pd.to_datetime(df["Actual_End"]) -
             pd.to_datetime(df["Scheduled_End"])).dt.total_seconds()
    if unit=="minutes": delta /= 60
    elif unit=="hours": delta /= 3600
    res = delta.groupby(df[gcol]).mean().reset_index(name=f"avg_delay_{unit}")
    return res

# ========== 新增 1) graph_export ==========
def graph_export(cur, args=None):
    """
    Convert current DataFrame (or full CSV) to a simple bipartite graph:
    Machine_ID ↔ Job_ID.  Saves as GEXF and returns file path.
    args = { "file": "output/plant_graph.gexf" }
    """
    import networkx as nx, os
    df = _df(cur)
    G = nx.Graph()
    for _, row in df.iterrows():
        m = row["Machine_ID"]; j = row["Job_ID"]
        G.add_node(m, bipartite="machine")
        G.add_node(j, bipartite="job")
        G.add_edge(m, j,
                   operation=row["Operation_Type"],
                   start=row["Scheduled_Start"],
                   end=row["Scheduled_End"])
    dst = args.get("file", "output/plant_graph.gexf") if args else "output/plant_graph.gexf"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    nx.write_gexf(G, dst)
    return dst

# ========== 新增 2) plot_machine_avg_bar ==========
def plot_machine_avg_bar(cur, args):
    """
    Bar chart of average <metric> per Machine_ID.
      args = { "metric": "Processing_Time",
               "file": "output/machine_avg.png" }
    """
    import matplotlib.pyplot as plt, os
    df = _df(cur)
    metric = args.get("metric", "Processing_Time")
    dst = args.get("file", f"output/avg_{metric}.png")
    grp = df.groupby("Machine_ID")[metric].mean().sort_values()
    plt.figure()
    grp.plot.bar()
    plt.ylabel(f"Average {metric}")
    plt.title(f"Average {metric} per Machine")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    plt.tight_layout()
    plt.savefig(dst)
    plt.close()
    return dst

# ========== 新增 3) plot_concurrent_tasks_line ==========
def plot_concurrent_tasks_line(cur, args=None):
    """
    Line chart: number of concurrent jobs vs. time.
      args = { "file": "output/concurrent_tasks.png",
               "freq": "10T" }   # resample frequency 10 minutes
    """
    import matplotlib.pyplot as plt, os
    df = _df(cur)
    freq = (args or {}).get("freq", "10T")
    dst  = (args or {}).get("file", "output/concurrent_tasks.png")
    # 建立每条任务的时间区间
    timeline = []
    for _, row in df.iterrows():
        s = pd.to_datetime(row["Actual_Start"])
        e = pd.to_datetime(row["Actual_End"])
        timeline.append((s, +1))   # start +1
        timeline.append((e, -1))   # end   -1
    tl = pd.DataFrame(timeline, columns=["time", "delta"])
    tl.set_index("time", inplace=True)
    # 按时间累积并重采样
    series = tl.sort_index()["delta"].cumsum().resample(freq).mean().fillna(method="ffill")
    plt.figure()
    series.plot()
    plt.ylabel("Concurrent Jobs")
    plt.title(f"Concurrent Jobs over Time (resample={freq})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    plt.savefig(dst)
    plt.close()
    return dst