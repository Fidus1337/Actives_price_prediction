import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.mode.chained_assignment = None

import os
import sys
import json
import traceback
from dotenv import load_dotenv

from Logs.LoggingSystem.LoggingSystem import LoggingSystem
from Classic_ml_model_solutions.PlotsBuilder.Plots_Builder import plot_roc, plot_metrics_vs_threshold, plot_confusion_matrix
from Classic_ml_model_solutions.Models_builder_pipeline.ModelsTrainer.range_model_trainer import range_model_train_pipeline
from Classic_ml_model_solutions.Models_builder_pipeline.ModelsTrainer.base_model_trainer import base_model_train_pipeline
from Classic_ml_model_solutions.Models_builder_pipeline.ModelsTrainer.ret_threshold_model_trainer import ret_threshold_model_train_pipeline
from Classic_ml_model_solutions.Models_builder_pipeline.ModelsTrainer.vol_scaled_model_trainer import vol_scaled_model_train_pipeline
from Classic_ml_model_solutions.Dataset_pipeline.SharedDataCache.SharedBaseDataCache import SharedBaseDataCache

load_dotenv("dev.env")

_api_key = os.getenv("COINGLASS_API_KEY")
if not _api_key:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")
API_KEY: str = _api_key

def load_config(config_path: str = "config.json") -> list:
    """Load configurations from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("runs", [])

def main_pipeline(cfg: dict, shared_cache: SharedBaseDataCache):
    """Run training pipeline for a single configuration.

    shared_cache already contains prepared data:
    raw -> normalize -> ffill -> date filter -> drop sparse -> engineered features -> TA.
    Here we only add the target, do final cleanup, and train.
    """
    base_feats = cfg["base_feats"]
    CONFIG_NAME = cfg["name"]
    threshold = cfg.get("threshold", 0.5)

    df_all = shared_cache.get_base_df()
    print(f"Data from cache. Shape: {df_all.shape}")

    models_folder = os.path.join("Classic_ml_model_solutions", "Created_models_to_use", CONFIG_NAME)
    os.makedirs(models_folder, exist_ok=True)

    # Select trainer by model type
    _TRAINERS = {
        "range_model": (range_model_train_pipeline, False),
        "ret_threshold_model": (ret_threshold_model_train_pipeline, True),
        "vol_scaled_model": (vol_scaled_model_train_pipeline, True),
        "base_model": (base_model_train_pipeline, True),
    }

    train_fn = None
    for model_type, (fn, use_roc) in _TRAINERS.items():
        if model_type in CONFIG_NAME:
            train_fn = fn
            break
    else:
        raise ValueError(f"Unknown model type in config name: {CONFIG_NAME}")

    print(f"\nTraining {CONFIG_NAME}...")
    res, _, _, oos_last = train_fn(
        df_all, base_feats, cfg, n_splits=4, thr=threshold
    )

    # --- OOS plots ---
    y, p = oos_last["y"].values, oos_last["p_up"].values
    n_samples = res["oos_full_metrics"]["n_oos_samples"]

    if use_roc:
        plot_roc(y, p, title=f"ROC ({CONFIG_NAME}, last fold OOS)", config_name=CONFIG_NAME)

    plot_metrics_vs_threshold(
        y, p,
        title=f"Metrics vs threshold ({CONFIG_NAME}, last fold OOS)",
        config_name=CONFIG_NAME
    )

    plot_confusion_matrix(
        y, p, threshold=threshold,
        title=f"Confusion Matrix ({CONFIG_NAME}, last fold OOS) ({n_samples} samples)",
        config_name=CONFIG_NAME
    )

def train_all_models_from_configs(config_in_project: str = "config.json", your_config=None):
    """Run main_pipeline() for each configuration."""
    configs = your_config if your_config is not None else load_config(config_in_project)

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
            traceback.print_exc()
            results[run_name] = f"FAILED: {e}"

    print("\n" + "=" * 60)
    print("SUMMARY:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    # Setup logging
    sys.stdout = LoggingSystem("Logs/logs.log")

    try:
        train_all_models_from_configs("configs/config.json")
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__