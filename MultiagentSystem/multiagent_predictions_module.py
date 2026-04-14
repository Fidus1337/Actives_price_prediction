import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from SharedDataCache.SharedBaseDataCache import SharedBaseDataCache
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter


def make_one_prediction(app, config: dict, forecast_start_date: str, cached_dataset: pd.DataFrame | None) -> dict:
    final_state = app.invoke({
        "config": config,
        "horizon": config["horizon"],
        "forecast_start_date": forecast_start_date,
        "agent_envolved_in_prediction": config["agent_envolved_in_prediction"],
        "cached_dataset": cached_dataset,
    })

    row = {
        "forecast_start_date": forecast_start_date,
        "y_predict": final_state.get("general_prediction_by_all_reports"),
        "y_predict_confidence": final_state.get("confidence_score"),
        "summary": final_state.get("general_reports_summary"),
        "reasoning": final_state.get("general_reports_reasoning"),
        "risks": final_state.get("general_reports_risks"),
    }

    # Flatten per-agent signals into columns: {agent_name}__prediction, __confidence
    for agent_name, signal in (final_state.get("agent_signals") or {}).items():
        short = agent_name.replace("agent_for_", "").replace("agent_for_analysing_", "")
        row[f"{short}__prediction"] = signal.get("prediction")
        row[f"{short}__confidence"] = signal.get("confidence")

    return row


def make_prediction_for_last_N_days(app, config: dict, last_days: int) -> pd.DataFrame:
    end_date = datetime.strptime(config["forecast_start_date"], "%Y-%m-%d")

    # Agents that require the CoinGlass base dataset
    _DATASET_AGENTS = {
        "agent_for_analysing_tech_indicators",
        "agent_for_analysing_onchain_indicators",
    }
    needs_dataset = bool(_DATASET_AGENTS & set(config.get("agent_envolved_in_prediction", [])))

    cached_dataset: pd.DataFrame | None = None
    if needs_dataset:
        api_key = os.environ["COINGLASS_API_KEY"]
        shared_cache = SharedBaseDataCache(api_key=api_key)
        cached_dataset = shared_cache.get_base_df()
        print(f"[predictions] Base dataset loaded: {cached_dataset.shape} "
              f"({cached_dataset['date'].min().date()} → {cached_dataset['date'].max().date()})")
    else:
        print("[predictions] No dataset-dependent agents — skipping CoinGlass fetch")

    rows = []
    for i in range(last_days):
        forecast_date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"[predictions] Day {i + 1}/{last_days} — forecast_date={forecast_date}")
        print(f"{'='*60}")

        print("DATE PREDICT:", forecast_date)
        row = make_one_prediction(app, config, forecast_date, cached_dataset)

        rows.append(row)

    return pd.DataFrame(rows)


def _fetch_bybit_btc_close_via_coinglass() -> pd.Series:
    """Fetch BTCUSDT daily close from CoinGlass /spot/price/history with exchange=Bybit.

    Single endpoint call — does not load the full SharedBaseDataCache pipeline.
    """
    api_key = os.environ["COINGLASS_API_KEY"]
    getter = FeaturesGetter(api_key=api_key)
    df = getter.get_history(
        endpoint_name="spot_price_history",
        exchange="Bybit",
        symbol="BTCUSDT",
        interval="1d",
        prefix="spot",
    )
    if df.empty:
        raise RuntimeError("CoinGlass returned empty spot_price_history for Bybit BTCUSDT")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date")["spot__close"].astype(float).sort_index()


def add_y_true(
    df: pd.DataFrame,
    horizon: int,
    close_col: str = "spot_price_history__close",
) -> pd.DataFrame:
    """Добавляет колонку y_true: реальное направление BTC через horizon дней.

    Источник цен — Bybit BTCUSDT spot через CoinGlass (один эндпоинт
    /spot/price/history, без полного SharedBaseDataCache).
    """
    if df.empty:
        df = df.copy()
        df["y_true"] = None
        return df

    try:
        close = _fetch_bybit_btc_close_via_coinglass()
    except Exception as exc:
        print(f"[add_y_true] Failed to fetch Bybit prices via CoinGlass: {exc}")
        print("[add_y_true] y_true will be None for all rows")
        df = df.copy()
        df["y_true"] = None
        return df

    y_true = []
    for _, row in df.iterrows():
        forecast_date = pd.Timestamp(row["forecast_start_date"]).normalize()
        target_date = forecast_date + timedelta(days=horizon)

        price_now = close.loc[forecast_date] if forecast_date in close.index else None
        price_then = close.loc[target_date] if target_date in close.index else None

        if price_now is None or price_then is None:
            y_true.append(None)
        else:
            y_true.append("LONG" if float(price_then) > float(price_now) else "SHORT")

    df = df.copy()
    df["y_true"] = y_true
    return df


def build_confusion_matrix(results_df: pd.DataFrame, horizon: int, output_path: Path) -> None:
    """Compare predicted LONG/SHORT against actual BTC price movement.

    Pulls BTCUSDT spot daily close from Bybit (via add_y_true) and for each
    forecast_start_date checks whether close[date + horizon] > close[date].
    Saves a confusion matrix plot to output_path.
    """
    # Если y_true ещё не посчитан — считаем на месте
    if "y_true" not in results_df.columns:
        results_df = add_y_true(results_df, horizon)

    # Оставляем только строки с валидными прогнозом и y_true
    valid = results_df[
        results_df["y_predict"].isin(["LONG", "SHORT"]) &
        results_df["y_true"].isin(["LONG", "SHORT"])
    ]

    if valid.empty:
        print("[confusion_matrix] Not enough matched dates to build confusion matrix")
        return

    actuals     = valid["y_true"].tolist()      # true_y: реальное направление BTC
    predictions = valid["y_predict"].tolist()   # predict_y: прогноз мультиагентной системы

    # --- Строим матрицу ошибок: строки = true_y, столбцы = predict_y ---
    labels = ["LONG", "SHORT"]
    cm = confusion_matrix(actuals, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # --- Рисуем и сохраняем график ---
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    accuracy = sum(a == p for a, p in zip(actuals, predictions)) / len(actuals)
    ax.set_title(f"Multiagent predictions  |  horizon={horizon}d  |  n={len(actuals)}  |  acc={accuracy:.1%}")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[confusion_matrix] Saved → {output_path}  (n={len(actuals)}, acc={accuracy:.1%})")