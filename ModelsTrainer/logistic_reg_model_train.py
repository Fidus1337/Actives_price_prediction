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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def oos_predictions_logreg(
    df: pd.DataFrame,
    features: list[str],
    target: str = "y_up_1d",
    n_splits: int = 5,
    model=None,
):
    """
    Out-of-sample predictions for Logistic Regression.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    features : list[str]
        List of feature column names
    target : str
        Target column name
    n_splits : int
        Number of splits for TimeSeriesSplit (used only if model is None)
    model : Pipeline or None
        Pre-trained model. If provided, predictions are made on the entire dataset
        without retraining. If None, walk-forward CV is performed.

    Returns:
    --------
    tuple : (out_df, auc, acc)
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date", kind="stable").reset_index(drop=True)

    X = d[features].copy()
    y = pd.to_numeric(d[target], errors="coerce")

    m = y.notna() & X.notna().any(axis=1)
    X = X.loc[m].reset_index(drop=True)
    y = y.loc[m].astype(int).reset_index(drop=True)
    dates = d.loc[m, "date"].reset_index(drop=True)

    # Use pre-trained model for predictions on entire dataset
    proba_oos = model.predict_proba(X)[:, 1]
    fold_id = np.zeros(len(X), dtype=int)  # all same fold when using pre-trained model

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
        tuple: (results_dict, best_model, oos_df) where:
            - results_dict: metrics from CV
            - best_model: pipeline from the fold with the best score
            - oos_df: DataFrame with OOS predictions from ALL folds (date, y, p_up, fold)
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date", kind="stable").reset_index(drop=True)

    X = d[features].copy()
    y = pd.to_numeric(d[target], errors="coerce")

    m = y.notna() & X.notna().any(axis=1)
    X, y = X.loc[m].reset_index(drop=True), y.loc[m].astype(int).reset_index(drop=True)
    dates = d.loc[m, "date"].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    accs, aucs, precs, recs, f1s = [], [], [], [], []
    models = []

    # Собираем OOS-предсказания со всех фолдов
    proba_oos = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        models.append(copy.deepcopy(pipe))

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)

        # Сохраняем OOS-предсказания
        proba_oos[test_idx] = proba
        fold_id[test_idx] = fold_i

        accs.append(accuracy_score(y_test, pred))
        precs.append(precision_score(y_test, pred, zero_division=0))
        recs.append(recall_score(y_test, pred, zero_division=0))
        f1s.append(f1_score(y_test, pred, zero_division=0))

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
        "f1": f1s,
    }
    scores = metric_map.get(best_metric, aucs)
    best_idx = int(np.nanargmax(scores))
    best_model = models[best_idx]

    # Формируем OOS DataFrame со всех фолдов
    oos_df = pd.DataFrame({
        "date": dates,
        "y": y,
        "p_up": proba_oos,
        "fold": fold_id,
    }).dropna(subset=["p_up"]).reset_index(drop=True)

    # --- Полный OOS для лучшей модели ---
    # Всё, что НЕ входит в train лучшего фолда — честный OOS
    all_splits = list(tscv.split(X))
    best_train_idx = all_splits[best_idx][0]

    oos_mask = np.ones(len(X), dtype=bool)
    oos_mask[best_train_idx] = False

    X_oos_full = X.loc[oos_mask].reset_index(drop=True)
    y_oos_full = y.loc[oos_mask].reset_index(drop=True)
    dates_oos_full = dates.loc[oos_mask].reset_index(drop=True)

    proba_oos_full = best_model.predict_proba(X_oos_full)[:, 1]
    pred_oos_full = (proba_oos_full >= thr).astype(int)

    oos_full_df = pd.DataFrame({
        "date": dates_oos_full,
        "y": y_oos_full,
        "p_up": proba_oos_full,
    }).reset_index(drop=True)

    oos_full_metrics = {
        "auc": float(roc_auc_score(y_oos_full, proba_oos_full)) if len(y_oos_full.unique()) == 2 else None,
        "acc": float(accuracy_score(y_oos_full, pred_oos_full)),
        "precision": float(precision_score(y_oos_full, pred_oos_full, zero_division=0)),
        "recall": float(recall_score(y_oos_full, pred_oos_full, zero_division=0)),
        "f1": float(f1_score(y_oos_full, pred_oos_full, zero_division=0)),
        "n_oos_samples": int(len(y_oos_full)),
    }

    results = {
        "n_features": len(features),
        "thr": thr,
        "best_metric": best_metric,
        "best_fold_idx": best_idx + 1,  # +1 т.к. fold_i начинается с 1
        "auc": float(aucs[best_idx]) if not np.isnan(aucs[best_idx]) else None,
        "acc": float(accs[best_idx]),
        "precision": float(precs[best_idx]),
        "recall": float(recs[best_idx]),
        "f1": float(f1s[best_idx]),
        "oos_full_metrics": oos_full_metrics,
    }

    return results, best_model, oos_df, oos_full_df

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
