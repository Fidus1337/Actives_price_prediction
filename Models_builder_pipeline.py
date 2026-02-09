import warnings
# Отключаем предупреждения, чтобы не засорять вывод
warnings.filterwarnings('ignore')
import pandas as pd
pd.options.mode.chained_assignment = None  # Отключаем SettingWithCopyWarning

import os
import sys
import json
import ta
from dotenv import load_dotenv
from fastapi import Body

# Ваши кастомные модули
from LoggingSystem.LoggingSystem import LoggingSystem
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from get_features_from_API import get_features
from FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from graphics_builder import plot_roc, plot_metrics_vs_threshold, print_threshold_analysis, plot_confusion_matrix
from ModelsTrainer.range_model_trainer import range_model_train_pipeline
from ModelsTrainer.base_model_trainer import base_model_train_pipeline
from ModelsTrainer.logistic_reg_model_train import add_lags

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

def add_ta_features_for_asset(df: pd.DataFrame, prefix: str, volume_col_override: str | None = None) -> pd.DataFrame:
    """Добавляет TA-индикаторы для актива с заданным префиксом.

    Parameters:
        prefix: префикс колонок актива (e.g. "gold", "sp500", "spot_price_history")
        volume_col_override: полное имя volume-колонки, если оно не {prefix}__volume
                             (e.g. "spot_price_history__volume_usd" для BTC)
    """
    df = df.copy()

    required = ['open', 'close', 'high', 'low', 'volume']
    col_map = {col: f"{prefix}__{col}" for col in required}

    # Позволяем переопределить имя volume-колонки
    if volume_col_override:
        col_map['volume'] = volume_col_override

    missing = [col_map[c] for c in required if col_map[c] not in df.columns]
    if missing:
        print(f"  Пропущены колонки для {prefix}: {missing}")
        return df

    temp_df = pd.DataFrame({
        'open': df[col_map['open']].values,
        'high': df[col_map['high']].values,
        'low': df[col_map['low']].values,
        'close': df[col_map['close']].values,
        'volume': df[col_map['volume']].values
    })

    temp_with_ta = ta.add_all_ta_features(
        temp_df,
        open="open", high="high", low="low", close="close", volume="volume",
        fillna=False
    )

    original_cols = {'open', 'high', 'low', 'close', 'volume'}
    ta_cols = [c for c in temp_with_ta.columns if c not in original_cols]

    for col in ta_cols:
        df.loc[df.index, f"{prefix}__{col}"] = temp_with_ta[col].values

    print(f"  Добавлено {len(ta_cols)} TA-фичей для {prefix}")
    return df

def main_pipeline(cfg: dict, api_key: str):
    """Основной пайплайн для одной конфигурации."""
    
    # 1. Распаковка конфига
    N_DAYS = cfg["N_DAYS"]
    base_feats = cfg["base_feats"]
    CONFIG_NAME = cfg["name"]
    TARGET_COLUMN_NAME = f"y_up_{N_DAYS}d"
    threshold = cfg.get("threshold", 0.5)

    getter = FeaturesGetter(api_key=api_key)
    features_engineer = FeaturesEngineer()

    # =============================================================================
    # 2. Сбор данных
    # =============================================================================
    print("Gathering features from API...")
    dfs = get_features(getter, api_key)
    df_all = merge_by_date(dfs, how="outer", dedupe="last")
    df_all = df_all.sort_values('date').reset_index(drop=True)
    print(f"Features gathered. Shape: {df_all.shape}")

    # =============================================================================
    # 3. Нормализация и первичное заполнение (ffill)
    # =============================================================================
    print("=" * 60)
    print("Normalizing spot columns & Applying ffill...")
    df_all = features_engineer.ensure_spot_prefix(df_all)
    
    feature_cols = [c for c in df_all.columns if c != "date"]
    df_all[feature_cols] = df_all[feature_cols].ffill()
    print(f"  Remaining NaN after ffill: {df_all[feature_cols].isna().sum().sum()}")

    # =============================================================================
    # 4. Генерация фичей и лагов (ВАЖНО: До обрезки даты!)
    # =============================================================================
    print("=" * 60)
    print("Engineering features & Adding lags...")
    
    # Инженерные фичи
    print(f"  Shape before feature engineering: {df_all.shape}")
    df_all = features_engineer.add_engineered_features(df_all, horizon=N_DAYS)
    
    df_all = add_ta_features_for_asset(df_all, prefix="gold")
    df_all = add_ta_features_for_asset(df_all, prefix="sp500")
    df_all = add_ta_features_for_asset(df_all, prefix="spot_price_history",
                                        volume_col_override="spot_price_history__volume_usd")

    # Лаги
    gold_cols = [c for c in df_all.columns if c.startswith("gold__") and "__lag" not in c]
    sp500_cols = [c for c in df_all.columns if c.startswith("sp500__") and "__lag" not in c]
    external_market_cols = gold_cols + sp500_cols
    EXTERNAL_LAGS = (1, 3, 5, 7, 10, 15)
    
    if external_market_cols:
        df_all = add_lags(df_all, cols=external_market_cols, lags=EXTERNAL_LAGS)
        print(f"  Added {len(external_market_cols) * len(EXTERNAL_LAGS)} lag features")
    
    # Целевая колонка
    df_all = features_engineer.add_y_up_custom(df_all, horizon=N_DAYS, close_col="spot_price_history__close")

    # =============================================================================
    # 5. Фильтрация по дате (1250 дней)
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
        
        res_rngp, model_rngp, oos_rngp = range_model_train_pipeline(
            df2, base_feats, cfg, n_splits=4, thr=threshold, choose_model_by_metric="f1"
        )
        
        print(f"Range model metrics (fold {res_rngp['best_fold_idx']}, by {res_rngp['best_metric']}): "
              f"AUC={res_rngp['auc']:.4f}, "
              f"Precision={res_rngp['precision']:.4f}, "
              f"Recall={res_rngp['recall']:.4f}, "
              f"F1={res_rngp['f1']:.4f}")

        # Графики Range
        y_rng, p_rng = oos_rngp["y"].values, oos_rngp["p_up"].values
        results_range = plot_metrics_vs_threshold(
            y_rng, p_rng,
            title=f"Metrics vs threshold (RANGE, OOF) - N{N_DAYS}_ma{ma_window}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_range, model_name=f"RANGE (N{N_DAYS}_ma{ma_window})")

        best_fold_rng = res_rngp["best_fold_idx"]
        oos_best_rng = oos_rngp[oos_rngp["fold"] == best_fold_rng]
        plot_confusion_matrix(
            oos_best_rng["y"].values, oos_best_rng["p_up"].values,
            threshold=threshold,
            title=f"Confusion Matrix (RANGE) - N{N_DAYS}_ma{ma_window} (fold {best_fold_rng})",
            config_name=CONFIG_NAME
        )

    # --- BASE MODEL ---
    elif "base_model" in CONFIG_NAME:
        print(f"\nTraining Logistic Regression (BASE: {TARGET_COLUMN_NAME})...")
        res_base, model_base, oos_df = base_model_train_pipeline(
            df2, base_feats, cfg, n_splits=4, thr=threshold, best_metric="f1"
        )

        # Графики Base
        y_b, p_b = oos_df["y"].values, oos_df["p_up"].values
        plot_roc(y_b, p_b, title="ROC (BASE, OOS)", config_name=CONFIG_NAME)

        results_base = plot_metrics_vs_threshold(
            y_b, p_b,
            title=f"Metrics vs threshold (BASE, OOF) - {TARGET_COLUMN_NAME}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_base, model_name=f"BASE ({TARGET_COLUMN_NAME})")

        best_fold_base = res_base["best_fold_idx"]
        oos_best_base = oos_df[oos_df["fold"] == best_fold_base]
        plot_confusion_matrix(
            oos_best_base["y"].values, oos_best_base["p_up"].values,
            threshold=threshold,
            title=f"Confusion Matrix (BASE) - {TARGET_COLUMN_NAME} (fold {best_fold_base})",
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