from dotenv import load_dotenv
import os
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
from quality_metrics import oos_proba_logreg, plot_roc

base_feats = [
    "spot_price_history__close__pct1",
    "spot_price_history__close__diff1",
    "futures_open_interest_aggregated_history__close__pct1",
    "futures_liquidation_aggregated_history__aggregated_short_liquidation_usd__diff1",
    "futures_global_long_short_account_ratio_history__global_account_long_percent__pct1"
    "futures_top_long_short_account_ratio_history__top_account_long_short_ratio__pct1"
    "premium__diff1",
    "cb_premium_abs"
]

def add_coverage(effect_tbl: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    eff = effect_tbl.copy()
    eff["coverage"] = eff["feature"].apply(lambda c: float(df[c].notna().mean()) if c in df.columns else 0.0)
    return eff.sort_values(["abs_cohen_d", "coverage"], ascending=[False, False]).reset_index(drop=True)

load_dotenv("dev.env")
API_KEY = os.getenv("COINGLASS_API_KEY")

if not API_KEY:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")

# HORIZON
N_DAYS = 1
TARGET_COLUMN_NAME = f"y_up_{N_DAYS}d"



if __name__ == "__main__":

    getter = FeaturesGetter(api_key=API_KEY)
    features_engineer = FeaturesEngineer()
    analyzer = CorrelationsAnalyzer()

    ## DATA GATHERING / PREPROCESSING
    # Собираем фичи в один датасет    
    dfs = get_features(getter, API_KEY)
    df_all = merge_by_date(dfs, how="outer", dedupe="last")

    ## FEATURE ENGINEERING
    # Нормализация спот-колонок
    df0 = features_engineer.ensure_spot_prefix(df_all)
    # Добавляем целевую колонку
    df1 = features_engineer.add_y_up_custom(df0, horizon=N_DAYS, close_col="spot_price_history__close")
    # Добавляем инженерные фичи 
    df2 = features_engineer.add_engineered_features(df1)
    # Удаляем строки с NA
    df1 = df1.dropna(subset=[TARGET_COLUMN_NAME, "spot_price_history__close"]).reset_index(drop=True)
    
    ## Calculating diff/pct + imbalance for every features
    df2 = features_engineer.add_engineered_features(df1, horizon=N_DAYS)

    ## CORRELATIONS ANALYSIS
    correlations_analyzer = CorrelationsAnalyzer()
    pear = correlations_analyzer.corr_report(df2, method="pearson", min_n=60, target_column_name=TARGET_COLUMN_NAME)
    spear = correlations_analyzer.corr_report(df2, method="spearman", min_n=60, target_column_name=TARGET_COLUMN_NAME)

    # filtering nan values at traget column
    df_ml = df_all.copy()
    df_ml["date"] = pd.to_datetime(df_ml["date"], errors="coerce")
    df_ml = df_ml.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df_ml["close"], errors="coerce")
    df_ml[TARGET_COLUMN_NAME] = (close.shift(-1*N_DAYS) > close).astype("Int64")
    df_ml = df_ml.dropna(subset=[TARGET_COLUMN_NAME, "close"]).reset_index(drop=True)

    corr_p = correlations_analyzer.corr_table_with_pvalues(df_ml, method="pearson", target=TARGET_COLUMN_NAME)
    corr_s = correlations_analyzer.corr_table_with_pvalues(df_ml, method="spearman",target=TARGET_COLUMN_NAME)

    effect_tbl = correlations_analyzer.group_effect_report(df2, target=TARGET_COLUMN_NAME, top_n=30)

    ## FILTERING FEATURES BY COVERAGE
    effect_tbl_cov = add_coverage(effect_tbl, df2)   # df2 = датафрейм с y_up_1d и engineered features
    good_features = effect_tbl_cov.query("coverage >= 0.85")["feature"].tolist()

    ## TRAINING LOGISTIC REGRESSION MODEL
    HIGH_COL  = "spot_price_history__high"
    LOW_COL   = "spot_price_history__low"
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

    res_rngp = walk_forward_logreg(d_rngp, features=feat_set, target=target_col, n_splits=5, thr=0.5)

    base_feats = [c for c in base_feats if c in df2.columns]

    # какие из них деривативные (кроме спота) — на них делаем лаги
    deriv = [c for c in base_feats if not c.startswith("spot_price_history__")]

    # Просто поменяй список цифр здесь
    my_lags = (1, 2, 3, 5, 10, 15)

    df_lag = add_lags(df2, cols=deriv, lags=my_lags)

    # Генератор списка фичей (чтобы не писать руками 100 строк)
    lag_feats = base_feats.copy()
    for lag in my_lags:
        lag_feats += [f"{c}__lag{lag}" for c in deriv]

    # Оставляем только те, что реально создались
    lag_feats = [c for c in lag_feats if c in df_lag.columns]

    res_base = walk_forward_logreg(df2, base_feats, n_splits=5, thr=0.5, target=TARGET_COLUMN_NAME)
    res_lag  = walk_forward_logreg(df_lag, lag_feats, n_splits=5, thr=0.5, target=TARGET_COLUMN_NAME)

    top_base, all_base = tune_logreg_timecv(df2, base_feats, target=TARGET_COLUMN_NAME, n_splits=5, score="auc", topk=10)
    top_lag,  all_lag  = tune_logreg_timecv(df_lag, lag_feats, target=TARGET_COLUMN_NAME, n_splits=5, score="auc", topk=10)

    # если хочешь сохранить лучшую конфигурацию:
    best = top_lag.iloc[0].to_dict() if len(top_lag) else top_base.iloc[0].to_dict()

        # ----- run -----
    oos, auc, acc = oos_predictions_logreg(df2, features=base_feats, n_splits=5, target=TARGET_COLUMN_NAME)

    # можно посмотреть по фолдам
    by_fold = oos.groupby("fold").apply(lambda g: pd.Series({
        "auc": roc_auc_score(g["y"], g["p_up"]) if g["y"].nunique()==2 else np.nan,
        "acc": accuracy_score(g["y"], (g["p_up"]>=0.5).astype(int)),
        "n": len(g)
    })).reset_index()

    tmp = oos.copy()
    tmp["bin"] = pd.qcut(tmp["p_up"], q=10, duplicates="drop")
    lift = tmp.groupby("bin").apply(lambda g: pd.Series({
        "n": len(g),
        "avg_p": g["p_up"].mean(),
        "win_if_long": float((g["y"]==1).mean()),
    })).reset_index()

    y_b, p_b = oos_proba_logreg(df2, base_feats, target=TARGET_COLUMN_NAME, n_splits=5)
    auc_b = plot_roc(y_b, p_b, title="ROC (BASE, OOS)")

    y_l, p_l = oos_proba_logreg(df_lag, lag_feats, target=TARGET_COLUMN_NAME, n_splits=5)
    auc_l = plot_roc(y_l, p_l, title="ROC (LAG, OOS)")

    print("AUC BASE:", auc_b)
    print("AUC LAG :", auc_l)


    




