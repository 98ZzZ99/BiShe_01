# rag_algorithms.py
"""
统一管理要跑的异常检测模型
EIF      – isotree.IsolationForest
LOF      – PyOD kNN‑LOF
COPOD    – PyOD COPOD
INNE     – PyOD INNE
OCSVM    – PyOD One‑ClassSVM
"""

# --- ① 依赖导入
from typing import Dict, Callable
import numpy as np
from isotree import IsolationForest          # Extended IF (EIF)
from pyod.models.lof import LOF              # kNN/LOF
from pyod.models.copod import COPOD          # COPOD
from pyod.models.ocsvm import OCSVM          # One‑Class SVM
from pyod.models.inne import INNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# --- ② 工厂函数 —— 返回已经 .fit() 好且带 .decision_scores_ 属性的对象
def _wrap_pyod(cls, **kw):
    """把 PyOD 模型封装成 “无参构造器” 便于延迟实例化"""
    return lambda: cls(**kw)

ALGOS: Dict[str, Callable[[], object]] = {
    "EIF":   lambda : IsolationForest(ntrees=300, sample_size='auto', ndim=1, nthreads=-1),
    "LOF":   lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
    "COPOD": _wrap_pyod(COPOD),
    "INNE":  _wrap_pyod(INNE, n_estimators=200, max_samples=256),
    "OCSVM": _wrap_pyod(OCSVM, kernel="rbf", nu=0.05, gamma="scale"),
}

def _try_get_score(model, X, prefer_neg=True) -> np.ndarray:
    """
    按优先级依次尝试获取连续得分:
      1. decision_scores_  (pyod)
      2. anomaly_score_    (isotree / pae-IF)
      3. decision_function (sklearn, output +1/-1 或连续)
      4. score_samples     (sklearn)
      5. predict           (±1 标签 -> 加极小噪声)
    返回时统一做符号翻转(若 prefer_neg=True，则把“值越小越异常”翻转过来)。
    """
    if hasattr(model, "decision_scores_"):
        scores = model.decision_scores_
    elif hasattr(model, "anomaly_score_"):
        scores = model.anomaly_score_
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X).ravel()
    elif hasattr(model, "score_samples"):
        scores = model.score_samples(X).ravel()
    else:
        # 最差的情况只能给 ±1 标签
        scores = model.predict(X).astype(float)
        # LOF/IF 等是 -1 异常 +1 正常，这里翻转一下方向
        prefer_neg = True

    # ----------------------------------------
    # 统一方向：越大 → 越异常
    if prefer_neg:
        scores = -scores
    return scores.astype(float)

# 供子图调用
请问一下这3种run_algo的写法有何区别？
def run_algo(name:str, X):
    """fit + decision_function → 返回 anomaly_score 1‑D ndarray"""
    mdl = ALGOS[name]()             # 现在才真正实例化
    mdl.fit(X)
    # PyOD / isotree 接口有差异：统一用 decision_function / predict
    if hasattr(mdl, "decision_function"):
        return mdl.decision_function(X)
    else:                            # e.g. isotree
        return mdl.predict(X, output="score")
def run_algo(name: str, X: np.ndarray) -> np.ndarray:
    if name == "EIF":
        scaler = StandardScaler()          # ★ 1. 标准化
        X_ = scaler.fit_transform(X)
    else:
        X_ = X                             # 其他算法用原数据

    model = ALGOS[name]()
    model.fit(X_)
    return -model.predict(X_).astype(float)
def run_algo(name: str, X: np.ndarray) -> np.ndarray:
    """
    统一调度各算法，返回连续 anomaly score。
    分数越大 → 越异常。
    """
    # 1) 特征缩放：距离 / 密度模型敏感
    if name in {"LOF", "OCSVM", "INNE", "COPOD"}:
        X_ = StandardScaler().fit_transform(X)
    else:                       # EIF 或其它
        X_ = X

    # 2) 训练
    model = ALGOS[name]()
    model.fit(X_)

    # 3) 取分
    scores = _try_get_score(model, X_, prefer_neg=True)

    # 4) 可选: 归一化 (解除注释即可)
    # scores = (scores - scores.mean()) / (scores.std(ddof=0) + 1e-9)

    return scores
