import os
import json
import joblib
import numpy as np
import pandas as pd
from Classic_ml_model_solutions.Models_builder_pipeline.ModelsTrainer.logistic_reg_model_train import walk_forward_logreg


def make_upNd_target_vol_scaled(
    close: pd.Series,
    horizon: int = 7,
    vol_window: int = 30,
    k: float = 0.5,
) -> pd.Series:
    close = close.astype(float)
    ret = np.log(close).diff()
    vol = ret.rolling(vol_window).std(ddof=0)

    fwd = np.log(close.shift(-horizon) / close)
    thr = k * vol * np.sqrt(horizon)

    # Keep target undefined where vol-based threshold (or future return) is undefined.
    valid = fwd.notna() & thr.notna()
    y = pd.Series(np.nan, index=close.index, dtype=float)
    y.loc[valid] = (fwd.loc[valid] >= thr.loc[valid]).astype(float)
    y.iloc[-horizon:] = np.nan
    return y


def vol_scaled_model_train_pipeline(
    df: pd.DataFrame,
    base_feats: list[str],
    cfg: dict,
    n_splits: int = 5,
    thr: float = 0.5,
) -> tuple[dict, object, pd.DataFrame, pd.DataFrame]:
    """
    Тренирует и сохраняет vol_scaled модель.

    Модель предсказывает: вырастет ли цена на >= k * vol * sqrt(horizon) через N дней.
    В отличие от base модели, порог адаптируется к текущей волатильности.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с подготовленными фичами (из SharedBaseDataCache)
    base_feats : list[str]
        Список базовых фичей
    cfg : dict
        Конфигурация с ключами: name, N_DAYS, vol_window, k
    n_splits : int
        Количество фолдов для walk-forward валидации
    thr : float
        Порог классификации

    Returns:
    --------
    tuple : (results_dict, trained_model, oos_df, oos_last_df)
    """
    N_DAYS = cfg["N_DAYS"]
    CONFIG_NAME = cfg["name"]
    vol_window = cfg.get("vol_window", 30)
    k = cfg.get("k", 0.5)

    CLOSE_COL = "spot_price_history__close"
    TARGET_COLUMN_NAME = f"y_up_vol_scaled_{N_DAYS}d"

    target_series = make_upNd_target_vol_scaled(
        close=df[CLOSE_COL],
        horizon=N_DAYS,
        vol_window=vol_window,
        k=k,
    )
    df_vs = df.copy()
    df_vs[TARGET_COLUMN_NAME] = target_series.values

    # Диагностика таргета
    valid_mask = df_vs[TARGET_COLUMN_NAME].notna()
    class_counts = (
        df_vs.loc[valid_mask, TARGET_COLUMN_NAME]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    class_share = (
        df_vs.loc[valid_mask, TARGET_COLUMN_NAME]
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )
    print(
        f"Target diagnostics: total={len(df_vs)}, valid={int(valid_mask.sum())}, "
        f"class_counts={class_counts}, class_share={class_share}"
    )

    df_vs = df_vs.dropna(subset=[TARGET_COLUMN_NAME]).reset_index(drop=True)

    if len(class_share) < 2:
        raise ValueError(
            f"Target {TARGET_COLUMN_NAME} has only one class after cleanup: {class_counts}. "
            "Training cannot proceed."
        )

    min_class_share = min(class_share.values())
    if min_class_share < 0.20:
        print(
            f"WARNING: Strong class imbalance detected for {TARGET_COLUMN_NAME} "
            f"(min class share={min_class_share:.3f}). "
            "Consider adjusting k or vol_window for this horizon."
        )

    feat_set = [c for c in base_feats if c in df_vs.columns]

    if not feat_set:
        missing = [c for c in base_feats if c not in df_vs.columns]
        raise ValueError(
            f"No features available for {CONFIG_NAME}. "
            f"Missing features (removed by NaN filter?): {missing[:5]}..."
        )

    print(f"Using {len(feat_set)}/{len(base_feats)} features: {feat_set[:3]}...")

    results, model, oos_df, oos_last_df = walk_forward_logreg(
        df_vs,
        features=feat_set,
        target=TARGET_COLUMN_NAME,
        n_splits=n_splits,
        thr=thr,
        purge_gap=N_DAYS,
    )

    models_folder = os.path.join("Classic_ml_model_solutions", "Created_models_to_use", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, f"model_vol_scaled_{CONFIG_NAME}.joblib")
    joblib.dump(model, model_path)

    oos_full = results["oos_full_metrics"]
    cv_avg = results["cv_avg_metrics"]
    metrics = {
        "config_name": CONFIG_NAME,
        "model_path": model_path,
        "target": TARGET_COLUMN_NAME,
        "vol_window": vol_window,
        "k": k,
        "features": feat_set,
        "n_features": results["n_features"],
        "thr": results["thr"],
        "eval_fold_idx": results["eval_fold_idx"],
        "auc": oos_full["auc"],
        "acc": oos_full["acc"],
        "precision": oos_full["precision"],
        "recall": oos_full["recall"],
        "f1": oos_full["f1"],
        "n_oos_samples": oos_full["n_oos_samples"],
        "cv_avg_auc": cv_avg["auc"],
        "cv_avg_acc": cv_avg["acc"],
        "cv_avg_precision": cv_avg["precision"],
        "cv_avg_recall": cv_avg["recall"],
        "cv_avg_f1": cv_avg["f1"],
    }
    metrics_path = os.path.join(models_folder, f"metrics_vol_scaled_{CONFIG_NAME}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Vol scaled model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Last fold OOS metrics (fold {results['eval_fold_idx']}, {oos_full['n_oos_samples']} samples):")
    print(f"  AUC:       {oos_full['auc']:.4f}" if oos_full['auc'] else "  AUC:       N/A")
    print(f"  Accuracy:  {oos_full['acc']:.4f}")
    print(f"  Precision: {oos_full['precision']:.4f}")
    print(f"  Recall:    {oos_full['recall']:.4f}")
    print(f"  F1:        {oos_full['f1']:.4f}")

    return results, model, oos_df, oos_last_df
