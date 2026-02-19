import os
import json
import joblib
import pandas as pd
from ModelsTrainer.logistic_reg_model_train import walk_forward_logreg
from new_targets import make_up7d_target_ret_threshold


def ret_threshold_model_train_pipeline(
    df: pd.DataFrame,
    base_feats: list[str],
    cfg: dict,
    n_splits: int = 5,
    thr: float = 0.5,
) -> tuple[dict, object, pd.DataFrame]:
    """
    Тренирует и сохраняет ret_threshold модель.

    Модель предсказывает: вырастет ли цена на >= ret_thr% через N дней.
    В отличие от base модели, мелкие движения (шум) считаются за класс 0.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с подготовленными фичами (из SharedBaseDataCache)
    base_feats : list[str]
        Список базовых фичей
    cfg : dict
        Конфигурация с ключами: name, N_DAYS, ret_thr, use_log
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
    ret_thr = cfg.get("ret_thr", 0.02)
    use_log = cfg.get("use_log", True)

    CLOSE_COL = "spot_price_history__close"
    TARGET_COLUMN_NAME = f"y_up_ret_thr_{N_DAYS}d"

    # Создаём таргет
    target_series = make_up7d_target_ret_threshold(
        close=df[CLOSE_COL],
        horizon=N_DAYS,
        ret_thr=ret_thr,
        use_log=use_log,
    )
    df_rt = df.copy()
    df_rt[TARGET_COLUMN_NAME] = target_series.values

    # Убираем строки без таргета
    df_rt = df_rt.dropna(subset=[TARGET_COLUMN_NAME]).reset_index(drop=True)

    feat_set = [c for c in base_feats if c in df_rt.columns]

    if not feat_set:
        missing = [c for c in base_feats if c not in df_rt.columns]
        raise ValueError(
            f"No features available for {CONFIG_NAME}. "
            f"Missing features (removed by NaN filter?): {missing[:5]}..."
        )

    print(f"Using {len(feat_set)}/{len(base_feats)} features: {feat_set[:3]}...")

    results, model, oos_df, oos_last_df = walk_forward_logreg(
        df_rt,
        features=feat_set,
        target=TARGET_COLUMN_NAME,
        n_splits=n_splits,
        thr=thr,
    )

    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, f"model_ret_thr_{CONFIG_NAME}.joblib")
    joblib.dump(model, model_path)

    oos_full = results["oos_full_metrics"]
    cv_avg = results["cv_avg_metrics"]
    metrics = {
        "config_name": CONFIG_NAME,
        "model_path": model_path,
        "target": TARGET_COLUMN_NAME,
        "ret_thr": ret_thr,
        "use_log": use_log,
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
    metrics_path = os.path.join(models_folder, f"metrics_ret_thr_{CONFIG_NAME}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Ret threshold model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Last fold OOS metrics (fold {results['eval_fold_idx']}, {oos_full['n_oos_samples']} samples):")
    print(f"  AUC:       {oos_full['auc']:.4f}" if oos_full['auc'] else "  AUC:       N/A")
    print(f"  Accuracy:  {oos_full['acc']:.4f}")
    print(f"  Precision: {oos_full['precision']:.4f}")
    print(f"  Recall:    {oos_full['recall']:.4f}")
    print(f"  F1:        {oos_full['f1']:.4f}")

    return results, model, oos_df, oos_last_df
