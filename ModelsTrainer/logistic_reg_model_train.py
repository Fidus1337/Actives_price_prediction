import warnings
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd
import json
import time
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug-0ffe56.log"))
    payload = {
        "sessionId": "0ffe56",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

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
):
    """
    Walk-forward cross-validation for Logistic Regression.

    OOS evaluation uses the LAST fold (most training data, most recent period,
    no selection bias). The final model for production is trained on ALL data.

    Returns:
        tuple: (results_dict, final_model, oos_df, oos_last_df) where:
            - results_dict: metrics from CV + last-fold OOS
            - final_model: pipeline trained on ALL data (for production)
            - oos_df: DataFrame with OOS predictions from ALL folds
            - oos_last_df: DataFrame with OOS predictions from last fold only
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date", kind="stable").reset_index(drop=True)

    X = d[features].copy()
    y = pd.to_numeric(d[target], errors="coerce")

    m = y.notna() & X.notna().any(axis=1)
    X, y = X.loc[m].reset_index(drop=True), y.loc[m].astype(int).reset_index(drop=True)
    dates = d.loc[m, "date"].reset_index(drop=True)

    # region agent log
    _agent_debug_log(
        run_id="pre-fix",
        hypothesis_id="H1_H2_entry",
        location="ModelsTrainer/logistic_reg_model_train.py:walk_forward_logreg:entry",
        message="entry_dataset_target_balance",
        data={
            "target": target,
            "thr": float(thr),
            "n_samples": int(len(y)),
            "pos_share": float(y.mean()) if len(y) else None,
            "neg_share": float(1 - y.mean()) if len(y) else None,
            "n_features": int(len(features)),
        },
    )
    # endregion

    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    accs, aucs, precs, recs, f1s = [], [], [], [], []

    proba_oos = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)
        pred_lo = (proba >= 0.4).astype(int)

        proba_oos[test_idx] = proba
        fold_id[test_idx] = fold_i

        accs.append(accuracy_score(y_test, pred))
        precs.append(precision_score(y_test, pred, zero_division=0))
        recs.append(recall_score(y_test, pred, zero_division=0))
        f1s.append(f1_score(y_test, pred, zero_division=0))

        if len(np.unique(y_test)) == 2:
            aucs.append(roc_auc_score(y_test, proba))
        else:
            aucs.append(np.nan)

        # region agent log
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H1_H2_H3_fold",
            location="ModelsTrainer/logistic_reg_model_train.py:walk_forward_logreg:fold",
            message="fold_metrics_and_probability_profile",
            data={
                "fold": int(fold_i),
                "thr": float(thr),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "train_pos_share": float(y_train.mean()) if len(y_train) else None,
                "test_pos_share": float(y_test.mean()) if len(y_test) else None,
                "proba_mean": float(np.mean(proba)) if len(proba) else None,
                "proba_p50": float(np.quantile(proba, 0.5)) if len(proba) else None,
                "proba_p90": float(np.quantile(proba, 0.9)) if len(proba) else None,
                "pred_pos_rate_thr": float(np.mean(pred)) if len(pred) else None,
                "pred_pos_rate_040": float(np.mean(pred_lo)) if len(pred_lo) else None,
                "acc_thr": float(accuracy_score(y_test, pred)),
                "recall_thr": float(recall_score(y_test, pred, zero_division=0)),
                "f1_thr": float(f1_score(y_test, pred, zero_division=0)),
                "recall_040": float(recall_score(y_test, pred_lo, zero_division=0)),
                "f1_040": float(f1_score(y_test, pred_lo, zero_division=0)),
            },
        )
        # endregion

    # OOS DataFrame from all folds
    oos_df = pd.DataFrame({
        "date": dates,
        "y": y,
        "p_up": proba_oos,
        "fold": fold_id,
    }).dropna(subset=["p_up"]).reset_index(drop=True)

    # --- OOS from the LAST fold (no selection bias) ---
    last_idx = n_splits - 1
    all_splits = list(tscv.split(X))
    last_test_idx = all_splits[last_idx][1]

    X_oos_last = X.iloc[last_test_idx].reset_index(drop=True)
    y_oos_last = y.iloc[last_test_idx].reset_index(drop=True)
    dates_oos_last = dates.iloc[last_test_idx].reset_index(drop=True)

    proba_oos_last = proba_oos[last_test_idx]
    pred_oos_last = (proba_oos_last >= thr).astype(int)

    oos_last_df = pd.DataFrame({
        "date": dates_oos_last,
        "y": y_oos_last,
        "p_up": proba_oos_last,
    }).reset_index(drop=True)

    oos_last_metrics = {
        "auc": float(roc_auc_score(y_oos_last, proba_oos_last)) if len(y_oos_last.unique()) == 2 else None,
        "acc": float(accuracy_score(y_oos_last, pred_oos_last)),
        "precision": float(precision_score(y_oos_last, pred_oos_last, zero_division=0)),
        "recall": float(recall_score(y_oos_last, pred_oos_last, zero_division=0)),
        "f1": float(f1_score(y_oos_last, pred_oos_last, zero_division=0)),
        "n_oos_samples": int(len(y_oos_last)),
    }

    # --- Final model: last fold's model (metrics and model are consistent) ---
    last_train_idx = all_splits[last_idx][0]
    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    final_pipe.fit(X.iloc[last_train_idx], y.iloc[last_train_idx])

    results = {
        "n_features": len(features),
        "thr": thr,
        "eval_fold_idx": n_splits,
        "auc": float(aucs[last_idx]) if not np.isnan(aucs[last_idx]) else None,
        "acc": float(accs[last_idx]),
        "precision": float(precs[last_idx]),
        "recall": float(recs[last_idx]),
        "f1": float(f1s[last_idx]),
        "oos_full_metrics": oos_last_metrics,
        "cv_avg_metrics": {
            "auc": float(np.nanmean(aucs)),
            "acc": float(np.mean(accs)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
        },
    }

    # region agent log
    _agent_debug_log(
        run_id="pre-fix",
        hypothesis_id="H3_H4_exit",
        location="ModelsTrainer/logistic_reg_model_train.py:walk_forward_logreg:exit",
        message="final_last_fold_and_cv_summary",
        data={
            "thr": float(thr),
            "last_fold_auc": results["auc"],
            "last_fold_acc": results["acc"],
            "last_fold_recall": results["recall"],
            "last_fold_f1": results["f1"],
            "cv_avg_auc": results["cv_avg_metrics"]["auc"],
            "cv_avg_acc": results["cv_avg_metrics"]["acc"],
            "cv_avg_recall": results["cv_avg_metrics"]["recall"],
            "cv_avg_f1": results["cv_avg_metrics"]["f1"],
        },
    )
    # endregion

    return results, final_pipe, oos_df, oos_last_df

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


def add_price_vs_sma_target(
    df: pd.DataFrame,
    close_col: str,
    date_col: str = "date",
    ma_window: int = 14,
    horizon: int = 1,
    out_target_col: str | None = None,
) -> pd.DataFrame:
    """
    Таргет: будет ли close[t+N] выше SMA(close)[t].

    y = 1 если close[t+N] > SMA_ma_window(close)[t]
    y = 0 иначе

    Также добавляет фичу close_sma{ma_window} — SMA на текущий день (без leakage).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.sort_values(date_col, kind="stable").reset_index(drop=True)

    close = pd.to_numeric(d[close_col], errors="coerce")

    # SMA close на текущий день — фича, доступна при предсказании
    sma_col = f"close_sma{ma_window}"
    d[sma_col] = close.rolling(ma_window, min_periods=ma_window).mean()

    # Для таргета: сравниваем будущий close с текущей SMA
    future_close = close.shift(-horizon)
    current_sma = d[sma_col]

    if out_target_col is None:
        out_target_col = f"y_close_above_sma_today_ma{ma_window}_N{horizon}"

    d[out_target_col] = np.where(
        future_close.notna() & current_sma.notna(),
        (future_close > current_sma).astype(int),
        np.nan,
    )
    return d


def add_crossover_target(
    df: pd.DataFrame,
    close_col: str,
    date_col: str = "date",
    ma_window: int = 14,
    horizon: int = 1,
    out_target_col: str | None = None,
) -> pd.DataFrame:
    """
    Таргет: будет ли пересечение close и SMA через horizon дней.

    above_today    = close[t]          > SMA[t]
    above_tomorrow = close[t+horizon]  > SMA[t+horizon]

    y = 1 если above_today != above_tomorrow  (будет пересечение / crossover)
    y = 0 если above_today == above_tomorrow  (без пересечения)

    Также добавляет фичу close_above_sma{ma_window} — бинарный признак
    (close выше SMA на текущий день).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.sort_values(date_col, kind="stable").reset_index(drop=True)

    close = pd.to_numeric(d[close_col], errors="coerce")

    # SMA close на текущий день
    sma_col = f"close_sma{ma_window}"
    d[sma_col] = close.rolling(ma_window, min_periods=ma_window).mean()

    sma = d[sma_col]

    # above_today: close[t] > SMA[t]  (известно в момент предсказания)
    above_today = close > sma

    # above_future: close[t+horizon] > SMA[t+horizon]  (будущее — для таргета)
    future_close = close.shift(-horizon)
    future_sma = sma.shift(-horizon)
    above_future = future_close > future_sma

    # Бинарная фича: close выше SMA сейчас
    d[f"close_above_sma{ma_window}"] = np.where(
        sma.notna(), above_today.astype(int), np.nan
    )

    if out_target_col is None:
        out_target_col = f"y_crossover_ma{ma_window}_N{horizon}"

    # y = 1 если будет пересечение (above_today != above_future)
    d[out_target_col] = np.where(
        future_close.notna() & future_sma.notna() & sma.notna(),
        (above_today != above_future).astype(int),
        np.nan,
    )
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
