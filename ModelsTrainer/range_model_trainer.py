import os
import json
import joblib
import pandas as pd
from ModelsTrainer.logistic_reg_model_train import walk_forward_logreg, add_range_target


def range_model_train_pipeline(
    df: pd.DataFrame,
    base_feats: list[str],
    cfg: dict,
    n_splits: int = 5,
    thr: float = 0.5,
) -> tuple[dict, object, pd.DataFrame]:
    """
    Тренирует и сохраняет Range модель.

    Модель предсказывает: будет ли range (high-low)/close через N дней
    выше текущей скользящей средней MA(ma_window).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с подготовленными фичами
    base_feats : list[str]
        Список базовых фичей
    cfg : dict
        Конфигурация с ключами: name, N_DAYS, ma_window, range_feats
    n_splits : int
        Количество фолдов для walk-forward валидации
    thr : float
        Порог классификации

    Returns:
    --------
    tuple : (results_dict, trained_model, oos_df)
    """
    N_DAYS = cfg["N_DAYS"]
    ma_window = cfg.get("ma_window", 14)
    range_feats = cfg.get("range_feats", None)
    CONFIG_NAME = cfg["name"]

    HIGH_COL = "spot_price_history__high"
    LOW_COL = "spot_price_history__low"
    CLOSE_COL = "spot_price_history__close"

    # Добавляем range target
    df_range = add_range_target(
        df,
        high_col=HIGH_COL,
        low_col=LOW_COL,
        close_col=CLOSE_COL,
        ma_window=ma_window,
        horizon=N_DAYS,
        use_pct=True,
        baseline_shift=1,
    )

    target_col = f"y_range_up_range_pct_N{N_DAYS}_ma{ma_window}"

    # Дефолтные range фичи если не указаны
    if range_feats is None:
        range_feats = [
            "range_pct",
            f"range_pct_ma{ma_window}",
        ]

    # Собираем финальный набор фичей
    feat_set = [c for c in (base_feats + range_feats) if c in df_range.columns]

    # Обучаем модель
    results, model, oos_df = walk_forward_logreg(
        df_range,
        features=feat_set,
        target=target_col,
        n_splits=n_splits,
        thr=thr,
        best_metric="auc"
    )

    # Сохраняем модель
    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, f"model_range_{CONFIG_NAME}.joblib")
    joblib.dump(model, model_path)

    # Сохраняем метрики лучшей модели в JSON
    metrics = {
        "config_name": CONFIG_NAME,
        "model_path": model_path,
        "n_features": results["n_features"],
        "thr": results["thr"],
        "best_metric": results["best_metric"],
        "best_fold_idx": results["best_fold_idx"],
        "auc": results["auc"],
        "acc": results["acc"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
    }
    metrics_path = os.path.join(models_folder, f"metrics_range_{CONFIG_NAME}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Range model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Best model metrics (fold {results['best_fold_idx']}, by {results['best_metric']}):")
    print(f"  AUC:       {results['auc']:.4f}" if results['auc'] else "  AUC:       N/A")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")

    return results, model, oos_df
