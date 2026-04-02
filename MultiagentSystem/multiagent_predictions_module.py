from pathlib import Path
import os
import sys
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — avoids ft2font init failure on Windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from SharedDataCache.SharedBaseDataCache import SharedBaseDataCache


def _resolve_graph_app(app: Any | None) -> Any:
    if app is not None:
        return app

    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "app"):
        return getattr(main_module, "app")

    try:
        from .multiagent_system_main import app as imported_app
        return imported_app
    except Exception as exc:
        raise RuntimeError(
            "LangGraph app is not available. Pass compiled app explicitly."
        ) from exc


def _prepare_dataset(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    horizon = int(config["horizon"])
    cache = SharedBaseDataCache(api_key=os.environ["COINGLASS_API_KEY"])
    base_df = cache.get_base_df()
    dataset = FeaturesEngineer().add_y_up_custom(
        base_df, horizon=horizon, close_col="spot_price_history__close"
    )
    return base_df, dataset, horizon


def _direction_to_binary(direction: str | None) -> int | None:
    if direction == "LONG":
        return 1
    if direction == "SHORT":
        return 0
    return None


def _save_confusion_matrix(results_dataset: pd.DataFrame, horizon: int, title: str, cm_path: Path) -> None:
    valid = results_dataset.dropna(subset=["y_predictions", f"y_up_{horizon}d"])
    if len(valid) < 2:
        return

    y_true = valid[f"y_up_{horizon}d"].astype(int)
    y_pred = valid["y_predictions"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["LOWER (0)", "HIGHER (1)"])
    disp.plot()
    plt.title(title)
    plt.savefig(cm_path)
    plt.close()


def make_prediction_for_date(
    config: dict,
    forecast_date: str | datetime | date,
    app: Any | None = None,
    base_df: pd.DataFrame | None = None,
    dataset_with_target: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if base_df is None or dataset_with_target is None:
        base_df, dataset_with_target, horizon = _prepare_dataset(config)
    else:
        horizon = int(config["horizon"])

    graph_app = _resolve_graph_app(app)
    target_dt = pd.to_datetime(forecast_date)
    matched = dataset_with_target[dataset_with_target["date"] == target_dt]
    if matched.empty:
        raise ValueError(f"No rows for forecast_date={target_dt.date()} in prepared dataset.")

    row = matched.iloc[-1]
    final_state = graph_app.invoke(
        {
            "config": config,
            "cached_dataset": base_df,
            "horizon": horizon,
            "forecast_start_date": row["date"].date(),
            "retry_agents": [],
            "retry_counts": {},
            "agent_envolved_in_prediction": config["agent_envolved_in_prediction"],
        }
    )

    direction = final_state.get("general_prediction_by_all_reports")
    prediction = _direction_to_binary(direction)
    return {
        "date": row["date"],
        "horizon": horizon,
        "y_true": int(row[f"y_up_{horizon}d"]),
        "direction": direction,
        "y_prediction": prediction,
        "confidence_score": final_state.get("confidence_score", 0),
    }


def make_prediction_for_last_N_days(
    config: dict,
    N: int,
    app: Any | None = None,
    cm_path: Path | None = None,
) -> pd.DataFrame:
    base_df, dataset_with_target, horizon = _prepare_dataset(config)
    anchor = datetime.strptime(config["forecast_start_date"], "%Y-%m-%d")
    eligible = dataset_with_target[dataset_with_target["date"] <= anchor]

    results = eligible[["date", f"y_up_{horizon}d"]].tail(N).copy()
    results["y_predictions"] = None
    results["confidence_score"] = None
    if results.empty:
        return results

    graph_app = _resolve_graph_app(app)
    cm_file = cm_path or (Path(__file__).parent / "agents" / "tech_agent_confusion_matrix.png")
    cm_file.parent.mkdir(parents=True, exist_ok=True)

    for done_count, (idx, row) in enumerate(results.iterrows(), start=1):
        one_day = make_prediction_for_date(
            config=config,
            forecast_date=row["date"],
            app=graph_app,
            base_df=base_df,
            dataset_with_target=dataset_with_target,
        )
        results.at[idx, "y_predictions"] = one_day["y_prediction"]
        results.at[idx, "confidence_score"] = one_day["confidence_score"]

        if done_count % 10 == 0:
            _save_confusion_matrix(
                results_dataset=results,
                horizon=horizon,
                title=f"Confusion Matrix ({done_count}/{N} predictions, horizon={horizon}d)",
                cm_path=cm_file,
            )

    return results


def build_confusion_matrix(results_dataset: pd.DataFrame, N_last_dates: int, horizon: int, cm_path: Path) -> None:
    valid_count = len(results_dataset.dropna(subset=["y_predictions"]))
    _save_confusion_matrix(
        results_dataset=results_dataset,
        horizon=horizon,
        title=f"Final Confusion Matrix ({valid_count}/{N_last_dates} predictions, horizon={horizon}d)",
        cm_path=cm_path,
    )