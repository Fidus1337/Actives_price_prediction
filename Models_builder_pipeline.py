import warnings
# Отключаем предупреждения, чтобы не засорять вывод
warnings.filterwarnings('ignore')
import pandas as pd
pd.options.mode.chained_assignment = None  # Отключаем SettingWithCopyWarning

import os
import sys
import json
from dotenv import load_dotenv
from fastapi import Body

# Ваши кастомные модули
from LoggingSystem.LoggingSystem import LoggingSystem
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from graphics_builder import plot_roc, plot_metrics_vs_threshold, print_threshold_analysis, plot_confusion_matrix
from ModelsTrainer.range_model_trainer import range_model_train_pipeline
from ModelsTrainer.base_model_trainer import base_model_train_pipeline
from shared_data_cache import SharedBaseDataCache

# Загрузка переменных окружения
load_dotenv("dev.env")
_api_key = os.getenv("COINGLASS_API_KEY")

if not _api_key:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")

API_KEY: str = _api_key

def load_config(config_path: str = "config.json") -> list:
    """Загружает конфигурации из JSON файла."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("runs", [])

def add_coverage(effect_tbl: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    eff = effect_tbl.copy()
    eff["coverage"] = eff["feature"].apply(lambda c: float(df[c].notna().mean()) if c in df.columns else 0.0)
    return eff.sort_values(["abs_cohen_d", "coverage"], ascending=[False, False]).reset_index(drop=True)

def main_pipeline(cfg: dict, shared_cache: SharedBaseDataCache):
    """Основной пайплайн для одной конфигурации."""

    # 1. Распаковка конфига
    N_DAYS = cfg["N_DAYS"]
    base_feats = cfg["base_feats"]
    CONFIG_NAME = cfg["name"]
    TARGET_COLUMN_NAME = f"y_up_{N_DAYS}d"
    threshold = cfg.get("threshold", 0.5)

    features_engineer = FeaturesEngineer()

    # =============================================================================
    # 2-4. Base data from shared cache (fetch, merge, ffill, features, TA, lags)
    # =============================================================================
    df_all = shared_cache.get_base_df()
    print(f"Base data from shared cache. Shape: {df_all.shape}")

    # Целевая колонка
    df_all = features_engineer.add_y_up_custom(df_all, horizon=N_DAYS, close_col="spot_price_history__close")

    # =============================================================================
    # 5. Фильтрация по дате (1500) дней)
    # =============================================================================
    print("=" * 60)
    print("Filtering last 1500 days...")

    df_all['date'] = pd.to_datetime(df_all['date'])
    max_date = df_all['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=1500)
    
    rows_total = len(df_all)
    df_all = df_all[df_all['date'] >= cutoff_date]
    print(f"  Rows kept: {len(df_all)} (from {rows_total})")

    # =============================================================================
    # 6. Очистка колонок и строк
    # =============================================================================
    print("=" * 60)
    print("Cleaning up columns and rows...")

    # Удаляем строки, где нет таргета (это последние N дней будущего)
    df_all = df_all.dropna(subset=[TARGET_COLUMN_NAME])
    
    # Удаляем колонки с >30% NaN
    nan_threshold = 0.3
    nan_ratio = df_all.isna().mean()
    cols_to_drop = [
        c for c in nan_ratio[nan_ratio > nan_threshold].index
        if not c.startswith("y_up_")
    ]
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns with >30% NaN")
        df_all = df_all.drop(columns=cols_to_drop)
    
    # Финальная очистка оставшихся NaN
    rows_before_final = len(df_all)
    df_all = df_all.dropna().reset_index(drop=True)
    print(f"  Final Dropna: removed {rows_before_final - len(df_all)} rows. Final shape: {df_all.shape}")

    # =============================================================================
    # 7. Тренировка моделей
    # =============================================================================
    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)
    
    df2 = df_all  # Используем подготовленный датафрейм

    # --- RANGE MODEL ---
    if "range_model" in CONFIG_NAME:
        ma_window = cfg.get("ma_window")
        print(f"\nTraining Logistic Regression (Range Target, MA={ma_window})...")

        res_rngp, model_rngp, oos_rngp, oos_full_rng = range_model_train_pipeline(
            df2, base_feats, cfg, n_splits=4, thr=threshold, choose_model_by_metric="accuracy"
        )

        oos_full_m = res_rngp["oos_full_metrics"]
        print(f"Range model full OOS ({oos_full_m['n_oos_samples']} samples, fold {res_rngp['best_fold_idx']}): "
              f"AUC={oos_full_m['auc']:.4f}, "
              f"Precision={oos_full_m['precision']:.4f}, "
              f"Recall={oos_full_m['recall']:.4f}, "
              f"F1={oos_full_m['f1']:.4f}")

        # Графики Range — на полном OOS
        y_rng, p_rng = oos_full_rng["y"].values, oos_full_rng["p_up"].values
        results_range = plot_metrics_vs_threshold(
            y_rng, p_rng,
            title=f"Metrics vs threshold (RANGE, full OOS) - N{N_DAYS}_ma{ma_window}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_range, model_name=f"RANGE (N{N_DAYS}_ma{ma_window})")

        plot_confusion_matrix(
            y_rng, p_rng,
            threshold=threshold,
            title=f"Confusion Matrix (RANGE, full OOS) - N{N_DAYS}_ma{ma_window} ({oos_full_m['n_oos_samples']} samples)",
            config_name=CONFIG_NAME
        )

    # --- BASE MODEL ---
    elif "base_model" in CONFIG_NAME:
        print(f"\nTraining Logistic Regression (BASE: {TARGET_COLUMN_NAME})...")
        res_base, model_base, oos_df, oos_full_base = base_model_train_pipeline(
            df2, base_feats, cfg, n_splits=4, thr=threshold, best_metric="accuracy"
        )

        # Графики Base — на полном OOS
        y_b, p_b = oos_full_base["y"].values, oos_full_base["p_up"].values
        plot_roc(y_b, p_b, title="ROC (BASE, full OOS)", config_name=CONFIG_NAME)

        results_base = plot_metrics_vs_threshold(
            y_b, p_b,
            title=f"Metrics vs threshold (BASE, full OOS) - {TARGET_COLUMN_NAME}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_base, model_name=f"BASE ({TARGET_COLUMN_NAME})")

        oos_full_m = res_base["oos_full_metrics"]
        plot_confusion_matrix(
            y_b, p_b,
            threshold=threshold,
            title=f"Confusion Matrix (BASE, full OOS) - {TARGET_COLUMN_NAME} ({oos_full_m['n_oos_samples']} samples)",
            config_name=CONFIG_NAME
        )

def run_all_configs(config_in_project: str = "config.json", your_config = None):
    """Запускает main() для каждой конфигурации из config.json."""
    
    if your_config is not None:
        configs = your_config
    else:
        configs = load_config(config_in_project)
    
    if not configs:
        print("No configurations found in config.json")
        return
    
    print(f"Found {len(configs)} configurations to run")
    print("=" * 60)
    
    cache = SharedBaseDataCache(api_key=API_KEY)

    results = {}
    for i, cfg in enumerate(configs, 1):
        run_name = cfg.get("name", f"run_{i}")

        print(f"\n{'='*60}")
        print(f"Running config [{i}/{len(configs)}]: {run_name}")
        print(f"N_DAYS: {cfg['N_DAYS']}, MA_WINDOW: {cfg.get('ma_window', 14)}")
        print(f"Features count: {len(cfg['base_feats'])}")
        print("=" * 60)

        try:
            main_pipeline(cfg, cache)
            results[run_name] = "SUCCESS"
        except Exception as e:
            print(f"ERROR in {run_name}: {e}")
            import traceback
            traceback.print_exc() # Полезно для отладки
            results[run_name] = f"FAILED: {e}"
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    # Логирование
    sys.stdout = LoggingSystem("logs.log")
    try:
        run_all_configs("config.json")
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__