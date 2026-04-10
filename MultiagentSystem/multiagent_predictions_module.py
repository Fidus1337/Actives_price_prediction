from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import yfinance as yf


def make_one_prediction(app, config: dict, forecast_start_date: str) -> dict:
    final_state = app.invoke({
        "config": config,
        "horizon": config["horizon"],
        "forecast_start_date": forecast_start_date,
        "agent_envolved_in_prediction": config["agent_envolved_in_prediction"],
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

    rows = []
    for i in range(last_days):
        forecast_date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"[predictions] Day {i + 1}/{last_days} — forecast_date={forecast_date}")
        print(f"{'='*60}")

        print("DATE PREDICT:", forecast_date)
        row = make_one_prediction(app, config, forecast_date)
        
        rows.append(row)

    return pd.DataFrame(rows)


def add_y_true(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Добавляет колонку y_true: реальное направление BTC через horizon дней.

    Скачивает исторические цены BTC через yfinance и для каждой строки
    сравнивает close[forecast_date + horizon] с close[forecast_date].
    Строки без данных получают y_true = None.
    """
    dates = pd.to_datetime(df["forecast_start_date"])
    price_start = dates.min() - timedelta(days=5)
    price_end   = dates.max() + timedelta(days=horizon + 5)

    btc = yf.download("BTC-USD", start=price_start, end=price_end, auto_adjust=True, progress=False)["Close"].squeeze()
    btc.index = pd.to_datetime(btc.index).normalize()

    def nearest(dt):
        for offset in range(4):
            candidate = dt + timedelta(days=offset)
            if candidate in btc.index:
                return btc.loc[candidate]
        return None

    y_true = []
    for _, row in df.iterrows():
        forecast_date = pd.Timestamp(row["forecast_start_date"])
        target_date   = forecast_date + timedelta(days=horizon)
        price_now  = nearest(forecast_date)
        price_then = nearest(target_date)
        if price_now is None or price_then is None:
            y_true.append(None)
        else:
            y_true.append("LONG" if price_then > price_now else "SHORT")

    df = df.copy()
    df["y_true"] = y_true
    return df


def build_confusion_matrix(results_df: pd.DataFrame, horizon: int, output_path: Path) -> None:
    """Compare predicted LONG/SHORT against actual BTC price movement.

    Downloads BTC-USD prices via yfinance and for each forecast_start_date
    checks whether close[date + horizon] > close[date] (actual LONG or SHORT).
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