import os
import json
import joblib
import pandas as pd
from ModelsTrainer.logistic_reg_model_train import walk_forward_logreg


def base_model_train_pipeline(
    df: pd.DataFrame,
    base_feats: list[str],
    cfg: dict,
    n_splits: int = 5,
    thr: float = 0.5,
    best_metric: str = "auc",
) -> tuple[dict, object, pd.DataFrame]:
    """
    Тренирует и сохраняет BASE модель.

    Модель предсказывает: будет ли цена выше через N дней.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с подготовленными фичами
    base_feats : list[str]
        Список базовых фичей
    cfg : dict
        Конфигурация с ключами: name, N_DAYS
    n_splits : int
        Количество фолдов для walk-forward валидации
    thr : float
        Порог классификации
    best_metric : str
        Метрика для выбора лучшей модели ("auc", "acc", "precision", "recall", "f1")

    Returns:
    --------
    tuple : (results_dict, trained_model, oos_df)
    """
    N_DAYS = cfg["N_DAYS"]
    CONFIG_NAME = cfg["name"]
    TARGET_COLUMN_NAME = f"y_up_{N_DAYS}d"

    # Фильтруем фичи по наличию в df
    feat_set = [c for c in base_feats if c in df.columns]

    # Обучаем модель
    results, model, oos_df = walk_forward_logreg(
        df,
        features=feat_set,
        target=TARGET_COLUMN_NAME,
        n_splits=n_splits,
        thr=thr,
        best_metric=best_metric,
    )

    # Сохраняем модель
    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, f"model_base_{CONFIG_NAME}.joblib")
    joblib.dump(model, model_path)

    # Сохраняем метрики лучшей модели в JSON
    metrics = {
        "config_name": CONFIG_NAME,
        "model_path": model_path,
        "target": TARGET_COLUMN_NAME,
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
    metrics_path = os.path.join(models_folder, f"metrics_base_{CONFIG_NAME}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Base model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Best model metrics (fold {results['best_fold_idx']}, by {results['best_metric']}):")
    print(f"  AUC:       {results['auc']:.4f}" if results['auc'] else "  AUC:       N/A")
    print(f"  Accuracy:  {results['acc']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")

    return results, model, oos_df
