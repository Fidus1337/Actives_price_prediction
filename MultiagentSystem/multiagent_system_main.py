import json
import os
import sys
from pathlib import Path

from .multiagent_predictions_module import (
    add_y_true,
    build_confusion_matrix,
    make_prediction_for_last_N_days,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")

from .multiagent_graph import app

from Logs.LoggingSystem.LoggingSystem import LoggingSystem


if __name__ == "__main__":
    log_path = Path(__file__).parent / "logs.log"
    sys.stdout = LoggingSystem(str(log_path), mode="w")

    # Load multiagent system config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "multiagent_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    N_days = 100

    load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")
    os.environ["COINGLASS_API_KEY"]  # fail fast if key is missing

    results_dataset = make_prediction_for_last_N_days(app, config, N_days)
    results_dataset = add_y_true(results_dataset, config["horizon"])

    output_path = Path(__file__).parent / "predictions_results.csv"
    results_dataset[
        [
            "forecast_start_date",
            "y_predict",
            "y_predict_confidence",
            "start_date_price",
            "btc_bybit_close_price",
            "btc_bybit_high_price",
            "btc_bybit_low_price",
            "y_true",
        ]
    ].to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved → {output_path}")

    cm_path = Path(__file__).parent / "confusion_matrix.png"
    build_confusion_matrix(results_dataset, config["horizon"], cm_path)
