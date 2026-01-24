import warnings
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def oos_predictions_logreg(df: pd.DataFrame, features: list[str], target="y_up_1d", n_splits=5):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date", kind="stable").reset_index(drop=True)

    X = d[features].copy()
    y = pd.to_numeric(d[target], errors="coerce")

    m = y.notna() & X.notna().any(axis=1)
    X = X.loc[m].reset_index(drop=True)
    y = y.loc[m].astype(int).reset_index(drop=True)
    dates = d.loc[m, "date"].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    proba_oos = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1, dtype=int)

    for i, (tr, te) in enumerate(tscv.split(X), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        proba_oos[te] = p
        fold_id[te] = i

    out = pd.DataFrame({
        "date": dates,
        "y": y,
        "p_up": proba_oos,
        "fold": fold_id,
    }).dropna(subset=["p_up"]).reset_index(drop=True)

    # общие метрики
    auc = roc_auc_score(out["y"], out["p_up"])
    acc = accuracy_score(out["y"], (out["p_up"] >= 0.5).astype(int))
    return out, auc, acc

def threshold_report(oos: pd.DataFrame, up_thr=0.55, down_thr=0.45):
    # long if p>=up_thr, short if p<=down_thr, else no trade
    p = oos["p_up"]
    y = oos["y"]

    take_long = p >= up_thr
    take_short = p <= down_thr
    take = take_long | take_short

    correct = (take_long & (y == 1)) | (take_short & (y == 0))

    rep = {
        "up_thr": up_thr,
        "down_thr": down_thr,
        "coverage_trades": float(take.mean()),
        "winrate_on_trades": float(correct[take].mean()) if take.any() else np.nan,
        "n_trades": int(take.sum()),
        "n_long": int(take_long.sum()),
        "n_short": int(take_short.sum()),
        "avg_p_up_all": float(p.mean()),
        "avg_p_up_long": float(p[take_long].mean()) if take_long.any() else np.nan,
        "avg_p_up_short": float(p[take_short].mean()) if take_short.any() else np.nan,
    }
    return rep

def walk_forward_logreg(
    df: pd.DataFrame,
    features: list[str],
    target: str = "y_up_1d",
    n_splits: int = 5,
    thr: float = 0.5,
    best_metric: str = "auc",
):
    """
    Walk-forward cross-validation for Logistic Regression.

    Returns:
        tuple: (results_dict, best_model) where best_model is the pipeline
               from the fold with the best score according to best_metric.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date", kind="stable").reset_index(drop=True)

    X = d[features].copy()
    y = pd.to_numeric(d[target], errors="coerce")

    m = y.notna() & X.notna().any(axis=1)
    X, y = X.loc[m].reset_index(drop=True), y.loc[m].astype(int).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    accs, aucs, precs, recs = [], [], [], []
    models = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        models.append(copy.deepcopy(pipe))

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)

        accs.append(accuracy_score(y_test, pred))
        precs.append(precision_score(y_test, pred, zero_division=0))
        recs.append(recall_score(y_test, pred, zero_division=0))

        # ROC AUC: если один класс — ставим nan, но длину сохраняем
        if len(np.unique(y_test)) == 2:
            aucs.append(roc_auc_score(y_test, proba))
        else:
            aucs.append(np.nan)

    # Select best model by metric
    metric_map = {
        "auc": aucs,
        "acc": accs,
        "precision": precs,
        "recall": recs,
    }
    scores = metric_map.get(best_metric, aucs)
    best_idx = int(np.nanargmax(scores))
    best_model = models[best_idx]

    results = {
        "n_features": len(features),
        "thr": thr,
        "acc_mean": float(np.nanmean(accs)),
        "acc_splits": accs,
        "precision_mean": float(np.nanmean(precs)),
        "precision_splits": precs,
        "recall_mean": float(np.nanmean(recs)),
        "recall_splits": recs,
        "auc_mean": float(np.nanmean(aucs)),
        "auc_splits": aucs,
    }

    return results, best_model

# --------- CV eval (one config) ----------
def walk_forward_logreg_cfg(
    df: pd.DataFrame,
    features: list[str],
    target: str = "y_up_1d",
    n_splits: int = 5,
    thr: float = 0.5,
    imputer_strategy: str = "mean",          # "mean" | "median"
    C: float = 1.0,
    penalty: str = "l2",                     # "l1" | "l2" | "elasticnet"
    solver: str = "lbfgs",                   # lbfgs/liblinear/saga
    l1_ratio: float | None = None,           # only for elasticnet+saga
    class_weight: str | dict | None = "balanced",
):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date", kind="stable").reset_index(drop=True)

    X = d[features].copy()
    y = pd.to_numeric(d[target], errors="coerce")

    m = y.notna() & X.notna().any(axis=1)
    X, y = X.loc[m].reset_index(drop=True), y.loc[m].astype(int).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    clf_kwargs = dict(
        max_iter=4000,
        class_weight=class_weight,
        C=float(C),
        penalty=penalty,
        solver=solver,
    )
    if penalty == "elasticnet":
        clf_kwargs["l1_ratio"] = float(l1_ratio if l1_ratio is not None else 0.5)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**clf_kwargs)),
    ])

    accs, aucs, precs, recs = [], [], [], []
    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= thr).astype(int)

        accs.append(accuracy_score(y_te, pred))
        precs.append(precision_score(y_te, pred, zero_division=0))
        recs.append(recall_score(y_te, pred, zero_division=0))

        if y_te.nunique() == 2:
            aucs.append(roc_auc_score(y_te, proba))

    return {
        "thr": thr,
        "imputer": imputer_strategy,
        "C": float(C),
        "penalty": penalty,
        "solver": solver,
        "l1_ratio": None if penalty != "elasticnet" else float(clf_kwargs["l1_ratio"]),
        "n_features": len(features),
        "acc_mean": float(np.mean(accs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "auc_mean": float(np.mean(aucs)) if aucs else np.nan,
        "acc_splits": accs,
        "auc_splits": aucs,
    }

# --------- Simple grid search ----------
from tqdm import tqdm

def tune_logreg_timecv(
    df: pd.DataFrame,
    features: list[str],
    target: str = "y_up_1d",
    n_splits: int = 5,
    score: str = "auc",          # "auc" | "acc" | "precision" | "recall"
    topk: int = 10,
):
    # небольшая, но полезная сетка
    grid = []

    # L2 (обычно лучший дефолт)
    for imp in ["mean", "median"]:
        for thr in [0.5, 0.52, 0.55]:
            for C in [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:
                grid.append(dict(imputer_strategy=imp, thr=thr, C=C, penalty="l2", solver="lbfgs", l1_ratio=None))

    # L1 (saga)
    for imp in ["mean", "median"]:
        for thr in [0.5, 0.52, 0.55]:
            for C in [0.05, 0.1, 0.3, 1.0, 3.0]:
                grid.append(dict(imputer_strategy=imp, thr=thr, C=C, penalty="l1", solver="saga", l1_ratio=None))

    # ElasticNet (saga)
    for imp in ["mean", "median"]:
        for thr in [0.5, 0.52, 0.55]:
            for C in [0.05, 0.1, 0.3, 1.0, 3.0]:
                for l1r in [0.2, 0.5, 0.8]:
                    grid.append(dict(imputer_strategy=imp, thr=thr, C=C, penalty="elasticnet", solver="saga", l1_ratio=l1r))

    rows = []
    for cfg in tqdm(grid):
        try:
            r = walk_forward_logreg_cfg(
                df=df,
                features=features,
                target=target,
                n_splits=n_splits,
                thr=cfg["thr"],
                imputer_strategy=cfg["imputer_strategy"],
                C=cfg["C"],
                penalty=cfg["penalty"],
                solver=cfg["solver"],
                l1_ratio=cfg["l1_ratio"],
            )
            rows.append(r)
        except Exception as e:
            # некоторые комбинации могут падать (редко, но бывает) — просто пропускаем
            continue

    res = pd.DataFrame(rows)

    key = {
        "auc": "auc_mean",
        "acc": "acc_mean",
        "precision": "precision_mean",
        "recall": "recall_mean",
    }[score]

    res = res.sort_values(key, ascending=False).reset_index(drop=True)
    return res.head(topk), res

def print_metrics(res: dict, title: str = "RESULTS"):
    print(f"\n{title}")
    print(f"thr={res['thr']} | n_features={res['n_features']}")
    print(f"Precision: {res['precision_mean']:.4f} | splits: {np.round(res['precision_splits'], 4)}")
    print(f"Recall:    {res['recall_mean']:.4f} | splits: {np.round(res['recall_splits'], 4)}")
    print(f"ROC AUC:   {res['auc_mean']:.4f} | splits: {np.round(res['auc_splits'], 4)}")

# Range model

def add_range_target(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str | None = None,   # если хочешь дополнительно range_pct
    date_col: str = "date",
    ma_window: int = 14,
    horizon: int = 1,               # N дней вперед для таргета
    use_pct: bool = False,          # False: high-low, True: (high-low)/close
    baseline_shift: int = 1,        # 1 => SMA только по прошлым дням (без leakage)
    out_target_col: str | None = None,
) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.sort_values(date_col, kind="stable").reset_index(drop=True)

    hi = pd.to_numeric(d[high_col], errors="coerce")
    lo = pd.to_numeric(d[low_col], errors="coerce")
    rng = hi-lo

    d["range_abs"] = rng

    if close_col is not None:
        close = pd.to_numeric(d[close_col], errors="coerce")
        d["range_pct"] = rng / close
    else:
        d["range_pct"] = np.nan

    # выбираем, на чем строим таргет и baseline
    base_series = d["range_pct"] if use_pct else d["range_abs"]
    base_name = "range_pct" if use_pct else "range_abs"

    # baseline SMA(14) — только прошлое (shift=1)
    d[f"{base_name}_ma{ma_window}"] = (
        base_series.shift(baseline_shift).rolling(ma_window, min_periods=ma_window).mean()
    )

    # будущий range на горизонте N
    fut = base_series.shift(-horizon)

    # бинарный таргет: будущий range выше/ниже текущей SMA(14)
    y = np.where(
        fut.notna() & d[f"{base_name}_ma{ma_window}"].notna(),
        (fut > d[f"{base_name}_ma{ma_window}"]).astype(int),
        np.nan,
    )

    if out_target_col is None:
        out_target_col = f"y_range_up_{base_name}_N{horizon}_ma{ma_window}"

    d[out_target_col] = y
    return d

def add_lags(df: pd.DataFrame, cols: list[str], lags=(1, 2)) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    for c in cols:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce")
            for L in lags:
                out[f"{c}__lag{L}"] = x.shift(L)
    return out