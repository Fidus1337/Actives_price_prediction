from dotenv import load_dotenv
import os
import sys
import json
import numpy as np
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


load_dotenv("dev.env")
API_KEY = os.getenv("COINGLASS_API_KEY")

if not API_KEY:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")


def main(base_feats, N_DAYS, TARGET_COLUMN_NAME, API_KEY, CONFIG_NAME):

    getter = FeaturesGetter(api_key=API_KEY)
    features_engineer = FeaturesEngineer()
    analyzer = CorrelationsAnalyzer()

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

    ## CORRELATIONS ANALYSIS
    # print("Calculating correlations...")
    # correlations_analyzer = CorrelationsAnalyzer()
    # pear = correlations_analyzer.corr_report(df2, method="pearson", min_n=60, target_column_name=TARGET_COLUMN_NAME)
    # spear = correlations_analyzer.corr_report(df2, method="spearman", min_n=60, target_column_name=TARGET_COLUMN_NAME)
    # print("Correlations calculated.")

    # filtering nan values at target column
    # df_ml = df_all.copy()
    # df_ml["date"] = pd.to_datetime(df_ml["date"], errors="coerce")
    # df_ml = df_ml.sort_values("date").reset_index(drop=True)
    # close = pd.to_numeric(df_ml["close"], errors="coerce")
    # df_ml[TARGET_COLUMN_NAME] = (close.shift(-1 * N_DAYS) > close).astype("Int64")
    # df_ml = df_ml.dropna(subset=[TARGET_COLUMN_NAME, "close"]).reset_index(drop=True)

    # corr_p = correlations_analyzer.corr_table_with_pvalues(df_ml, method="pearson", target=TARGET_COLUMN_NAME)
    # corr_s = correlations_analyzer.corr_table_with_pvalues(df_ml, method="spearman", target=TARGET_COLUMN_NAME)

    # print("Calculating group effect report...")
    # effect_tbl = correlations_analyzer.group_effect_report(df2, target=TARGET_COLUMN_NAME, top_n=30)

    ## FILTERING FEATURES BY COVERAGE
    # print("Filtering features by coverage...")
    # effect_tbl_cov = add_coverage(effect_tbl, df2)
    # good_features = effect_tbl_cov.query("coverage >= 0.85")["feature"].tolist()
    # print(f"Features filtered. {len(good_features)} good features found.")

    ## TRAINING LOGISTIC REGRESSION MODEL ################
    ## Предсказывает будет ли range через N дней выше текущего MA?
    
    print("Training Logistic Regression (Range Target)...")
    HIGH_COL = "spot_price_history__high"
    LOW_COL = "spot_price_history__low"
    CLOSE_COL = "spot_price_history__close"

    d_rngp = add_range_target(
        df2,
        high_col=HIGH_COL,
        low_col=LOW_COL,
        close_col=CLOSE_COL,
        ma_window=14,
        horizon=N_DAYS,
        use_pct=True,      # (high-low)/close
        baseline_shift=1,
    )

    target_col = f"y_range_up_range_pct_N{N_DAYS}_ma14"

    range_feats = [
        "range_pct",
        "range_pct_ma14",
    ]
    feat_set = [c for c in (base_feats + range_feats) if c in d_rngp.columns]

    # Model for prediction trend
    res_rngp = walk_forward_logreg(d_rngp, features=feat_set, target=target_col, n_splits=5, thr=0.5)
    
    ###################################################
    
    base_feats = [c for c in base_feats if c in df2.columns]

    # какие из них деривативные (кроме спота) — на них делаем лаги
    deriv = [c for c in base_feats if not c.startswith("spot_price_history__")]

    # Просто поменяй список цифр здесь
    my_lags = (1, 2, 3, 5, 10)

    df_lag = add_lags(df2, cols=deriv, lags=my_lags)

    # Генератор списка фичей (чтобы не писать руками 100 строк)
    lag_feats = base_feats.copy()
    for lag in my_lags:
        lag_feats += [f"{c}__lag{lag}" for c in deriv]

    # Оставляем только те, что реально создались
    lag_feats = [c for c in lag_feats if c in df_lag.columns]

    print("Training Logistic Regression (BASE)...")
    res_base = walk_forward_logreg(df2, base_feats, n_splits=5, thr=0.5, target=TARGET_COLUMN_NAME)
    # print("Training Logistic Regression (LAG)...")
    # res_lag = walk_forward_logreg(df_lag, lag_feats, n_splits=5, thr=0.5, target=TARGET_COLUMN_NAME)

    print("Tuning Logistic Regression (BASE)...")
    top_base, all_base = tune_logreg_timecv(df2, base_feats, target=TARGET_COLUMN_NAME, n_splits=5, score="auc", topk=10)
    # print("Tuning Logistic Regression (LAG)...")
    # top_lag, all_lag = tune_logreg_timecv(df_lag, lag_feats, target=TARGET_COLUMN_NAME, n_splits=5, score="auc", topk=10)

    # если хочешь сохранить лучшую конфигурацию:
    # best = top_lag.iloc[0].to_dict() if len(top_lag) else top_base.iloc[0].to_dict()
    # print("Best config found.")

    # ----- run -----
    print("Running OOS predictions...")
    oos, auc, acc = oos_predictions_logreg(df2, features=base_feats, n_splits=5, target=TARGET_COLUMN_NAME)
    print(f"OOS predictions complete. AUC: {auc:.4f}, ACC: {acc:.4f}")

    # можно посмотреть по фолдам
    by_fold = oos.groupby("fold").apply(lambda g: pd.Series({
        "auc": roc_auc_score(g["y"], g["p_up"]) if g["y"].nunique() == 2 else np.nan,
        "acc": accuracy_score(g["y"], (g["p_up"] >= 0.5).astype(int)),
        "n": len(g)
    })).reset_index()

    tmp = oos.copy()
    tmp["bin"] = pd.qcut(tmp["p_up"], q=10, duplicates="drop")
    lift = tmp.groupby("bin").apply(lambda g: pd.Series({
        "n": len(g),
        "avg_p": g["p_up"].mean(),
        "win_if_long": float((g["y"] == 1).mean()),
    })).reset_index()

    y_b, p_b = oos_proba_logreg(df2, base_feats, target=TARGET_COLUMN_NAME, n_splits=5)
    auc_b = plot_roc(y_b, p_b, title="ROC (BASE, OOS)", config_name=CONFIG_NAME)
    
    # Анализ метрик по порогам для BASE модели
    results_base = plot_metrics_vs_threshold(
        y_b, p_b, 
        title=f"Metrics vs threshold (BASE, OOF) - {TARGET_COLUMN_NAME}",
        config_name=CONFIG_NAME
    )
    print_threshold_analysis(results_base, model_name=f"BASE ({TARGET_COLUMN_NAME})")

    # y_l, p_l = oos_proba_logreg(df_lag, lag_feats, target=TARGET_COLUMN_NAME, n_splits=5)
    # auc_l = plot_roc(y_l, p_l, title="ROC (LAG, OOS)", config_name=CONFIG_NAME)
    
    # Анализ метрик по порогам для LAG модели
    # results_lag = plot_metrics_vs_threshold(
    #     y_l, p_l, 
    #     title=f"Metrics vs threshold (LAG, OOF) - {TARGET_COLUMN_NAME}",
    #     config_name=CONFIG_NAME
    # )
    # print_threshold_analysis(results_lag, model_name=f"LAG ({TARGET_COLUMN_NAME})")

    print("AUC BASE:", auc_b)
    # print("AUC LAG :", auc_l)


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
        n_days = cfg["N_DAYS"]
        base_feats = cfg["base_feats"]
        config_name = cfg["name"]
        target_column_name = f"y_up_{n_days}d"
        
        print(f"\n{'='*60}")
        print(f"Running config [{i}/{len(configs)}]: {run_name}")
        print(f"N_DAYS: {n_days}, TARGET: {target_column_name}")
        print(f"Features count: {len(base_feats)}")
        print("=" * 60)
        
        try:
            main(base_feats, n_days, target_column_name, API_KEY, config_name)
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