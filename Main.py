from dotenv import load_dotenv
import os
import sys
import json
import numpy as np
import joblib
from LoggingSystem.LoggingSystem import LoggingSystem
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from get_features_from_API import get_features
from FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from CorrelationsAnalyzer.CorrelationsAnalyzer import CorrelationsAnalyzer
import pandas as pd
from logistic_reg_model_train import walk_forward_logreg, add_range_target, add_lags, tune_logreg_timecv, print_metrics, oos_predictions_logreg, threshold_report
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import brier_score_loss
from tqdm import tqdm
from quality_metrics import oos_proba_logreg, plot_roc, plot_metrics_vs_threshold, print_threshold_analysis


def load_config(config_path: str = "config.json") -> list:
    """Загружает конфигурации из JSON файла."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("runs", [])


def add_coverage(effect_tbl: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    eff = effect_tbl.copy()
    eff["coverage"] = eff["feature"].apply(lambda c: float(df[c].notna().mean()) if c in df.columns else 0.0)
    return eff.sort_values(["abs_cohen_d", "coverage"], ascending=[False, False]).reset_index(drop=True)


def train_estimate_range_model(
    df: pd.DataFrame,
    base_feats: list[str],
    cfg: dict,
    n_splits: int = 5,
    thr: float = 0.5,
) -> tuple[dict, object]:
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
    tuple : (results_dict, trained_model)
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
    results, model = walk_forward_logreg(
        df_range,
        features=feat_set,
        target=target_col,
        n_splits=n_splits,
        thr=thr
    )

    # Сохраняем модель
    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, f"model_range_{CONFIG_NAME}.joblib")
    joblib.dump(model, model_path)
    print(f"Range model saved to {model_path}")
    print(f"Range model AUC: {results['auc_mean']:.4f}")

    return results, model


load_dotenv("dev.env")
_api_key = os.getenv("COINGLASS_API_KEY")

if not _api_key:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")

API_KEY: str = _api_key

def main_pipeline(cfg: dict, api_key: str):
    """Основной пайплайн для одной конфигурации."""
    
    # Извлекаем параметры из конфига
    N_DAYS = cfg["N_DAYS"]
    base_feats = cfg["base_feats"]
    CONFIG_NAME = cfg["name"]
    TARGET_COLUMN_NAME = f"y_up_{N_DAYS}d"
    ma_window = cfg.get("ma_window", 14)
    range_feats = cfg.get("range_feats", None)

    # Чтобы получать с апишки 
    getter = FeaturesGetter(api_key=api_key)
    # Для формирования новых фичей
    features_engineer = FeaturesEngineer()
    # Для анализа корреляций признаков с целевым параметром
    # analyzer = CorrelationsAnalyzer()

    ## DATA GATHERING / PREPROCESSING
    # Собираем фичи в один датасет    
    print("Gathering features from API...")
    dfs = get_features(getter, API_KEY)
    df_all = merge_by_date(dfs, how="outer", dedupe="last")
    print(f"Features gathered. Shape: {df_all.shape}")

    ## FEATURE ENGINEERING
    # Нормализация спот-колонок
    print("Normalizing spot columns...")
    df0 = features_engineer.ensure_spot_prefix(df_all)
    
    # Добавляем целевую колонку
    print(f"Adding target column (horizon={N_DAYS}d)...")
    df1 = features_engineer.add_y_up_custom(df0, horizon=N_DAYS, close_col="spot_price_history__close")
    
    # Удаляем строки с NA в целевой колонке и close
    df1 = df1.dropna(subset=[TARGET_COLUMN_NAME, "spot_price_history__close"]).reset_index(drop=True)
    
    # Добавляем инженерные фичи (один раз)
    print("Adding engineered features...")
    df2 = features_engineer.add_engineered_features(df1, horizon=N_DAYS)
    print(f"Feature engineering complete. Shape: {df2.shape}")
    
    ### ТРЕНИРОВКА МОДЕЛЕЙ 
    # Создаём папку для моделей
    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)
    
    ## ТРЕНИРОВКА RANGE МОДЕЛИ
    print("Training Logistic Regression (Range Target)...")
    res_rngp, model_rngp = train_estimate_range_model(df2, base_feats, cfg, n_splits=5, thr=0.5)
    print(f"Range model AUC: {res_rngp['auc_mean']:.4f}")

    ## ТРЕНИРОВКА BASE МОДЕЛИ
    print("Training Logistic Regression (BASE)...")
    # Сохраняем фичи с конфига
    base_feats = [c for c in base_feats if c in df2.columns]
    res_base, model_base = walk_forward_logreg(df2, base_feats, n_splits=5, thr=0.5, target=TARGET_COLUMN_NAME)

    # Сохраняем BASE модель
    model_path = os.path.join(models_folder, f"model_base_{CONFIG_NAME}.joblib")
    joblib.dump(model_base, model_path)
    print(f"Base model saved to {model_path}")

    # Оцениваем качество BASE модели
    print("Running OOS predictions...")
    oos, auc, acc = oos_predictions_logreg(df2, features=base_feats, n_splits=5, target=TARGET_COLUMN_NAME)
    print(f"OOS predictions complete. AUC: {auc:.4f}, ACC: {acc:.4f}")

    y_b, p_b = oos_proba_logreg(df2, base_feats, target=TARGET_COLUMN_NAME, n_splits=5)
    auc_b = plot_roc(y_b, p_b, title="ROC (BASE, OOS)", config_name=CONFIG_NAME)
    
    # Анализ метрик по порогам для BASE модели
    results_base = plot_metrics_vs_threshold(
        y_b, p_b, 
        title=f"Metrics vs threshold (BASE, OOF) - {TARGET_COLUMN_NAME}",
        config_name=CONFIG_NAME
    )
    print_threshold_analysis(results_base, model_name=f"BASE ({TARGET_COLUMN_NAME})")

    print("AUC BASE:", auc_b)


def run_all_configs(config_path: str = "config.json"):
    """Запускает main() для каждой конфигурации из config.json."""
    configs = load_config(config_path)
    
    if not configs:
        print("No configurations found in config.json")
        return
    
    print(f"Found {len(configs)} configurations to run")
    print("=" * 60)
    
    results = {}
    for i, cfg in enumerate(configs, 1):
        run_name = cfg.get("name", f"run_{i}")

        print(f"\n{'='*60}")
        print(f"Running config [{i}/{len(configs)}]: {run_name}")
        print(f"N_DAYS: {cfg['N_DAYS']}, MA_WINDOW: {cfg.get('ma_window', 14)}")
        print(f"Features count: {len(cfg['base_feats'])}")
        print("=" * 60)

        try:
            main_pipeline(cfg, API_KEY)
            results[run_name] = "SUCCESS"
        except Exception as e:
            print(f"ERROR in {run_name}: {e}")
            results[run_name] = f"FAILED: {e}"
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    # Перенаправляем вывод в файл logs.log и консоль
    sys.stdout = LoggingSystem("logs.log")
    try:
        run_all_configs("config.json")
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__