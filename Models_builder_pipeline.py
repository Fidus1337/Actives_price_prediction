from dotenv import load_dotenv
import os
import sys
import json
from LoggingSystem.LoggingSystem import LoggingSystem
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from get_features_from_API import get_features
from FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
import pandas as pd
from graphics_builder import plot_roc, plot_metrics_vs_threshold, print_threshold_analysis, plot_confusion_matrix
from ModelsTrainer.range_model_trainer import range_model_train_pipeline
from ModelsTrainer.base_model_trainer import base_model_train_pipeline
from ModelsTrainer.logistic_reg_model_train import add_lags
import ta

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

def add_ta_features_for_asset(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Добавляет TA-индикаторы для актива с заданным префиксом."""
    df = df.copy()  # Работаем с копией
    
    required = ['open', 'close', 'high', 'low', 'volume']
    col_map = {col: f"{prefix}__{col}" for col in required}

    missing = [col_map[c] for c in required if col_map[c] not in df.columns]
    if missing:
        print(f"  Пропущены колонки для {prefix}: {missing}")
        return df

    # Создаём временный DataFrame с reset индексом
    temp_df = pd.DataFrame({
        'open': df[col_map['open']].values,
        'high': df[col_map['high']].values,
        'low': df[col_map['low']].values,
        'close': df[col_map['close']].values,
        'volume': df[col_map['volume']].values
    })

    # Добавляем TA-индикаторы
    temp_with_ta = ta.add_all_ta_features(
        temp_df,
        open="open", high="high", low="low", close="close", volume="volume",
        fillna=False
    )

    original_cols = {'open', 'high', 'low', 'close', 'volume'}
    ta_cols = [c for c in temp_with_ta.columns if c not in original_cols]

    # Присваиваем через .values с явным reset_index
    for col in ta_cols:
        df.loc[df.index, f"{prefix}__{col}"] = temp_with_ta[col].values

    print(f"  Добавлено {len(ta_cols)} TA-фичей для {prefix}")
    
    return df

def main_pipeline(cfg: dict, api_key: str):
    """Основной пайплайн для одной конфигурации."""
    
    # Извлекаем параметры из конфига
    N_DAYS = cfg["N_DAYS"]
    base_feats = cfg["base_feats"]
    CONFIG_NAME = cfg["name"]
    TARGET_COLUMN_NAME = f"y_up_{N_DAYS}d"
    range_feats = cfg.get("range_feats", None)
    threshold = cfg.get("threshold", 0.5)

    # Чтобы получать с апишки фичи
    getter = FeaturesGetter(api_key=api_key)
    # Для формирования новых фичей
    features_engineer = FeaturesEngineer()
    # Для анализа корреляций признаков с целевым параметром
    # analyzer = CorrelationsAnalyzer()

    ## DATA GATHERING / PREPROCESSING
    # Собираем фичи в один датасет
    print("Gathering features from API...")
    dfs = get_features(getter, api_key)
    df_all = merge_by_date(dfs, how="outer", dedupe="last")
    df_all = df_all.sort_values('date')
    print(f"Features gathered. Shape: {df_all.shape}")
    
    # Forward fill для заполнения пропущенных значений предыдущими
    print("Applying forward fill...")
    feature_cols = [c for c in df_all.columns if c != "date"]
    df_all[feature_cols] = df_all[feature_cols].ffill()
    print(f"Forward fill complete. Remaining NaN count: {df_all[feature_cols].isna().sum().sum()}")

    ## FEATURE ENGINEERING
    # Нормализация спот-колонок
    print("Normalizing spot columns...")
    df0 = features_engineer.ensure_spot_prefix(df_all)

    # Добавляем целевую колонку
    print(f"Adding target column (horizon={N_DAYS}d)...")
    df1 = features_engineer.add_y_up_custom(df0, horizon=N_DAYS, close_col="spot_price_history__close")
          
    # Добавляем инженерные фичи (один раз)
    print("Adding engineered features...")
    df2 = features_engineer.add_engineered_features(df1, horizon=N_DAYS)
    print(f"Feature engineering complete. Shape: {df2.shape}")

    # Добавляем лаговые признаки для Gold и S&P500
    EXTERNAL_LAGS = (1, 3, 5, 7, 10, 15)
    gold_cols = [c for c in df2.columns if c.startswith("gold__") and "__lag" not in c]
    sp500_cols = [c for c in df2.columns if c.startswith("sp500__") and "__lag" not in c]
    external_market_cols = gold_cols + sp500_cols

    if external_market_cols:
        print(f"Adding lag features for Gold ({len(gold_cols)}) and S&P500 ({len(sp500_cols)}) columns...")
        df2 = add_lags(df2, cols=external_market_cols, lags=EXTERNAL_LAGS)
        print(f"Added {len(external_market_cols) * len(EXTERNAL_LAGS)} lag features. New shape: {df2.shape}")

    df2 = df2.sort_values('date')
    
    # Удаляем колонки с >30% NaN значений (исключая target-колонки)
    # nan_threshold = 0.3
    # nan_ratio = df2.isna().mean()
    # cols_to_drop = [
    #     c for c in nan_ratio[nan_ratio > nan_threshold].index
    #     if not c.startswith("y_up_")
    # ]
    # if cols_to_drop:
    #     print(f"Dropping {len(cols_to_drop)} columns with >{nan_threshold*100:.0f}% NaN values:")
    #     for col in cols_to_drop:
    #         print(f"  - {col}: {nan_ratio[col]*100:.1f}% NaN")
    #     df2 = df2.drop(columns=cols_to_drop)

    # Удаляем строки с NaN в целевой колонке
    rows_before = len(df2)
    df2 = df2.dropna(subset=[TARGET_COLUMN_NAME]).reset_index(drop=True)
    rows_dropped = rows_before - len(df2)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with NaN in {TARGET_COLUMN_NAME}")
    
    df2 = df2.dropna()

    print("NAN-values::")
    print(df2.isna().sum())
    
    print("DF2:", df2.shape)
    
    ### ТРЕНИРОВКА МОДЕЛЕЙ
    # Создаём папку для моделей
    models_folder = os.path.join("Models", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)

    ## ТРЕНИРОВКА RANGE МОДЕЛИ
    if "range_model" in CONFIG_NAME:

        ma_window = cfg.get("ma_window")

        print("Training Logistic Regression (Range Target)...")
        res_rngp, model_rngp, oos_rngp = range_model_train_pipeline(df2, base_feats, cfg, n_splits=4, thr=threshold, choose_model_by_metric="f1")
        print(f"Range model metrics (fold {res_rngp['best_fold_idx']}, by {res_rngp['best_metric']}): "
            f"AUC={res_rngp['auc']:.4f}, "
            f"Precision={res_rngp['precision']:.4f}, "
            f"Recall={res_rngp['recall']:.4f}, "
            f"F1={res_rngp['f1']:.4f}")

        # # Анализ метрик по порогам для RANGE модели
        y_rng, p_rng = oos_rngp["y"].values, oos_rngp["p_up"].values
        results_range = plot_metrics_vs_threshold(
            y_rng, p_rng,
            title=f"Metrics vs threshold (RANGE, OOF) - N{N_DAYS}_ma{ma_window}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_range, model_name=f"RANGE (N{N_DAYS}_ma{ma_window})")

        # Confusion matrix для RANGE модели (только лучший фолд)
        best_fold_rng = res_rngp["best_fold_idx"]
        oos_best_rng = oos_rngp[oos_rngp["fold"] == best_fold_rng]
        plot_confusion_matrix(
            oos_best_rng["y"].values, oos_best_rng["p_up"].values,
            threshold=threshold,
            title=f"Confusion Matrix (RANGE) - N{N_DAYS}_ma{ma_window} (fold {best_fold_rng})",
            config_name=CONFIG_NAME
        )

    ## ТРЕНИРОВКА BASE МОДЕЛИ
    elif "base_model" in CONFIG_NAME:

        print("Training Logistic Regression (BASE)...")
        res_base, model_base, oos_df = base_model_train_pipeline(
            df2, base_feats, cfg, n_splits=4, thr=threshold, best_metric="f1"
        )

        # Графики
        y_b, p_b = oos_df["y"].values, oos_df["p_up"].values
        plot_roc(y_b, p_b, title="ROC (BASE, OOS)", config_name=CONFIG_NAME)

        # Анализ метрик по порогам для BASE модели
        results_base = plot_metrics_vs_threshold(
            y_b, p_b,
            title=f"Metrics vs threshold (BASE, OOF) - {TARGET_COLUMN_NAME}",
            config_name=CONFIG_NAME
        )
        print_threshold_analysis(results_base, model_name=f"BASE ({TARGET_COLUMN_NAME})")

        # Confusion matrix для BASE модели (только лучший фолд)
        best_fold_base = res_base["best_fold_idx"]
        oos_best_base = oos_df[oos_df["fold"] == best_fold_base]
        plot_confusion_matrix(
            oos_best_base["y"].values, oos_best_base["p_up"].values,
            threshold=threshold,
            title=f"Confusion Matrix (BASE) - {TARGET_COLUMN_NAME} (fold {best_fold_base})",
            config_name=CONFIG_NAME
        )


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