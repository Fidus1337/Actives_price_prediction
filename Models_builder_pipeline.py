import warnings
# Отключаем предупреждения, чтобы не засорять вывод
warnings.filterwarnings('ignore')
import pandas as pd
pd.options.mode.chained_assignment = None  # Отключаем SettingWithCopyWarning

import os
import sys
import json
from dotenv import load_dotenv


from LoggingSystem.LoggingSystem import LoggingSystem
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from graphics_builder import plot_roc, plot_metrics_vs_threshold, print_threshold_analysis, plot_confusion_matrix
from ModelsTrainer.range_model_trainer import range_model_train_pipeline
from ModelsTrainer.base_model_trainer import base_model_train_pipeline
from ModelsTrainer.ret_threshold_model_trainer import ret_threshold_model_train_pipeline
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

def main_pipeline(cfg: dict, shared_cache: SharedBaseDataCache):
    """Основной пайплайн для одной конфигурации.

    shared_cache уже содержит подготовленные данные:
    raw -> normalize -> ffill -> date filter -> drop sparse -> engineered features -> TA
    Здесь добавляем только target + финальная очистка + тренировка.
    """

    # 1. Распаковка конфига
    N_DAYS = cfg["N_DAYS"]
    base_feats = cfg["base_feats"]
    CONFIG_NAME = cfg["name"]
    TARGET_COLUMN_NAME = f"y_up_{N_DAYS}d"
    threshold = cfg.get("threshold", 0.5)

    features_engineer = FeaturesEngineer()

    # =============================================================================
    # 2. Base data from shared cache + target column
    # =============================================================================
    df_all = shared_cache.get_base_df()
    df_all = features_engineer.add_y_up_custom(df_all, horizon=N_DAYS, close_col="spot_price_history__close")
    print(f"Data from cache + target. Shape: {df_all.shape}")

    # =============================================================================
    # 3. Финальная очистка
    # =============================================================================
    # Удаляем строки, где нет таргета (последние N_DAYS дней)
    df_all = df_all.dropna(subset=[TARGET_COLUMN_NAME])

    # Финальная очистка оставшихся NaN (от TA lookback ~20 строк)
    rows_before = len(df_all)
    df_all = df_all.dropna().reset_index(drop=True)
    print(f"Cleanup: {rows_before} -> {len(df_all)} rows. Final shape: {df_all.shape}")

    # =============================================================================
    # 4. Тренировка моделей
    # =============================================================================
    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)

    # --- RANGE MODEL ---
    if "range_model" in CONFIG_NAME:
        ma_window = cfg.get("ma_window")
        print(f"\nTraining Logistic Regression (Range Target, MA={ma_window})...")

        res_rngp, model_rngp, oos_rngp, oos_last_rng = range_model_train_pipeline(
            df_all, base_feats, cfg, n_splits=4, thr=threshold
        )

        oos_full_m = res_rngp["oos_full_metrics"]
        print(f"Range model last fold OOS ({oos_full_m['n_oos_samples']} samples, fold {res_rngp['eval_fold_idx']}): "
              f"AUC={oos_full_m['auc']:.4f}, "
              f"Precision={oos_full_m['precision']:.4f}, "
              f"Recall={oos_full_m['recall']:.4f}, "
              f"F1={oos_full_m['f1']:.4f}")

        y_rng, p_rng = oos_last_rng["y"].values, oos_last_rng["p_up"].values
        results_range = plot_metrics_vs_threshold(
            y_rng, p_rng,
            title=f"Metrics vs threshold (RANGE, last fold OOS) - N{N_DAYS}_ma{ma_window}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_range, model_name=f"RANGE (N{N_DAYS}_ma{ma_window})")

        plot_confusion_matrix(
            y_rng, p_rng,
            threshold=threshold,
            title=f"Confusion Matrix (RANGE, last fold OOS) - N{N_DAYS}_ma{ma_window} ({oos_full_m['n_oos_samples']} samples)",
            config_name=CONFIG_NAME
        )

    # --- RET THRESHOLD MODEL ---
    elif "ret_threshold_model" in CONFIG_NAME:
        ret_thr = cfg.get("ret_thr", 0.02)
        print(f"\nTraining Logistic Regression (RET THRESHOLD: {TARGET_COLUMN_NAME}, ret_thr={ret_thr})...")

        res_rt, model_rt, oos_rt, oos_last_rt = ret_threshold_model_train_pipeline(
            df_all, base_feats, cfg, n_splits=4, thr=threshold
        )

        y_rt, p_rt = oos_last_rt["y"].values, oos_last_rt["p_up"].values
        plot_roc(y_rt, p_rt, title="ROC (RET_THR, last fold OOS)", config_name=CONFIG_NAME)

        results_rt = plot_metrics_vs_threshold(
            y_rt, p_rt,
            title=f"Metrics vs threshold (RET_THR, last fold OOS) - {N_DAYS}d ret_thr={ret_thr}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_rt, model_name=f"RET_THR ({N_DAYS}d, ret_thr={ret_thr})")

        oos_full_m = res_rt["oos_full_metrics"]
        plot_confusion_matrix(
            y_rt, p_rt,
            threshold=threshold,
            title=f"Confusion Matrix (RET_THR, last fold OOS) - {N_DAYS}d ret_thr={ret_thr} ({oos_full_m['n_oos_samples']} samples)",
            config_name=CONFIG_NAME
        )

    # --- BASE MODEL ---
    elif "base_model" in CONFIG_NAME:
        print(f"\nTraining Logistic Regression (BASE: {TARGET_COLUMN_NAME})...")
        res_base, model_base, oos_df, oos_last_base = base_model_train_pipeline(
            df_all, base_feats, cfg, n_splits=4, thr=threshold
        )

        y_b, p_b = oos_last_base["y"].values, oos_last_base["p_up"].values
        plot_roc(y_b, p_b, title="ROC (BASE, last fold OOS)", config_name=CONFIG_NAME)

        results_base = plot_metrics_vs_threshold(
            y_b, p_b,
            title=f"Metrics vs threshold (BASE, last fold OOS) - {TARGET_COLUMN_NAME}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_base, model_name=f"BASE ({TARGET_COLUMN_NAME})")

        oos_full_m = res_base["oos_full_metrics"]
        plot_confusion_matrix(
            y_b, p_b,
            threshold=threshold,
            title=f"Confusion Matrix (BASE, last fold OOS) - {TARGET_COLUMN_NAME} ({oos_full_m['n_oos_samples']} samples)",
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