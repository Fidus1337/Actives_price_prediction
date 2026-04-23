import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from Classic_ml_solutions_rework.Dataset_pipeline.get_features import FeaturesGetterRework

UNBIAS_URL = "https://unbias.fyi/api/v1/sentiment"
CONFIG_PATH = _PROJECT_ROOT / "configs" / "multiagent_config.json"


def get_authors_signals(author: str, days: int, key: str, asset: str = "BTC") -> dict:
    r = requests.get(
        UNBIAS_URL,
        params={"asset": asset, "days": days, "handle": author},
        headers={"X-API-Key": key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_author_df(author: str, days: int, key: str) -> pd.DataFrame:
    """Возвращает DataFrame [date, author, confidence] для одного автора."""
    payload = get_authors_signals(author, days, key)
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame(columns=["date", "author", "confidence"])

    df = pd.DataFrame(data)[["date", "sentiment_score"]].rename(
        columns={"sentiment_score": "confidence"}
    )
    df["date"] = pd.to_datetime(df["date"])
    df["author"] = author
    # на случай дублей по (author, date) оставляем последний
    df = df.drop_duplicates(subset=["date", "author"], keep="last")
    return df[["date", "author", "confidence"]]


def fetch_all_authors(
    authors: list[str], days: int, key: str, sleep_sec: float = 0.2
) -> pd.DataFrame:
    """Long-формат: [date, author, confidence] по всем авторам."""
    frames: list[pd.DataFrame] = []
    for a in authors:
        try:
            df = fetch_author_df(a, days, key)
            print(f"[ok]  {a:>18}  rows={len(df)}")
            frames.append(df)
        except Exception as e:
            print(f"[err] {a:>18}  {e}")
        time.sleep(sleep_sec)  # мягкий троттлинг
    if not frames:
        return pd.DataFrame(columns=["date", "author", "confidence"])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "author"])


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на дату: колонка на автора вида {author}__conf."""
    if df_long.empty:
        return pd.DataFrame(columns=["date"])
    wide = (
        df_long.pivot_table(index="date", columns="author", values="confidence", aggfunc="last")
               .sort_index()
               .reset_index()
    )
    wide.columns = ["date"] + [f"{c}__conf" for c in wide.columns[1:]]
    return wide


def aggregate_confidence(df_wide: pd.DataFrame, conf_suffix: str = "__conf") -> pd.DataFrame:
    """Среднее арифметическое сигналов авторов по дате, NaN игнорируются.

    Добавляет колонки:
      - confidence_mean: среднее по не-NaN авторам
      - voters_count:    сколько авторов проголосовало в этот день
    """
    author_cols = [c for c in df_wide.columns if c.endswith(conf_suffix)]
    out = df_wide.copy()
    out["confidence_mean"] = out[author_cols].mean(axis=1, skipna=True)
    out["voters_count"] = out[author_cols].notna().sum(axis=1).astype(int)
    out.loc[out["voters_count"] == 0, "confidence_mean"] = pd.NA
    return out


def build_confusion_matrix(
    df: pd.DataFrame,
    threshold: float = 50.0,
    conf_col: str = "confidence_mean",
    y_true_col: str = "y_true",
) -> tuple[pd.DataFrame, dict]:
    """Строит confusion matrix: confidence_mean > threshold -> LONG, < threshold -> SHORT.

    Возвращает (cm, metrics). Дни с NaN в любом из столбцов или equal-to-threshold отбрасываются.
    """
    mask = df[conf_col].notna() & df[y_true_col].notna() & (df[conf_col] != threshold)
    sub = df.loc[mask].copy()
    sub["y_pred"] = sub[conf_col].gt(threshold).map({True: "LONG", False: "SHORT"})

    labels = ["LONG", "SHORT"]
    cm = pd.crosstab(
        sub[y_true_col].astype(str),
        sub["y_pred"],
        rownames=["y_true"],
        colnames=["y_pred"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    tp = int(cm.loc["LONG", "LONG"])
    tn = int(cm.loc["SHORT", "SHORT"])
    fp = int(cm.loc["SHORT", "LONG"])
    fn = int(cm.loc["LONG", "SHORT"])
    total = tp + tn + fp + fn

    metrics = {
        "n": total,
        "accuracy":  (tp + tn) / total if total else float("nan"),
        "precision_long":  tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall_long":     tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision_short": tn / (tn + fn) if (tn + fn) else float("nan"),
        "recall_short":    tn / (tn + fp) if (tn + fp) else float("nan"),
    }
    return cm, metrics


def sweep_threshold(
    df: pd.DataFrame,
    thresholds: "list[float] | None" = None,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Перебирает пороги и возвращает таблицу метрик, отсортированную по `metric`."""
    if thresholds is None:
        import numpy as np
        thresholds = [float(x) for x in np.arange(30.0, 95.5, 0.5)]

    rows: list[dict] = []
    for thr in thresholds:
        _, m = build_confusion_matrix(df, threshold=float(thr))
        m["balanced"] = (m["recall_long"] + m["recall_short"]) / 2
        rows.append({"thr": float(thr), **m})
    res = pd.DataFrame(rows)
    return res.sort_values(metric, ascending=False).reset_index(drop=True)


def add_y_true(df: pd.DataFrame, horizon: int, price_col: str = "close") -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()
    future = out[price_col].shift(-horizon)
    out[f"{price_col}_future_{horizon}d"] = future
    y = pd.Series(pd.NA, index=out.index, dtype="object")
    y[future > out[price_col]] = "LONG"
    y[future < out[price_col]] = "SHORT"
    out["y_true"] = y
    return out


if __name__ == "__main__":
    load_dotenv("dev.env")
    key = os.getenv("UNBIAS_KEY")
    if not key:
        raise RuntimeError("UNBIAS_KEY is not set in dev.env")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    twitter_cfg = cfg["agent_settings"]["agent_for_twitter_analysis"]
    authors: list[str] = twitter_cfg["authors"]
    DAYS = 365
    HORIZON = int(cfg.get("horizon", 1))

    print(f"fetching {len(authors)} authors, days={DAYS}")
    df_long = fetch_all_authors(authors, days=DAYS, key=key)

    out_dir = Path(__file__).resolve().parent
    long_csv = out_dir / "unbias_signals_long.csv"
    df_long.to_csv(long_csv, index=False)
    print(f"\nsaved long  -> {long_csv}  shape={df_long.shape}")

    df_wide = long_to_wide(df_long)
    df_wide = aggregate_confidence(df_wide)
    wide_csv = out_dir / "unbias_signals_wide.csv"
    df_wide.to_csv(wide_csv, index=False)
    print(f"saved wide  -> {wide_csv}  shape={df_wide.shape}")
    print(f"voters_count distribution:\n{df_wide['voters_count'].value_counts().sort_index()}")

    getter = FeaturesGetterRework(env_path=_PROJECT_ROOT / "dev.env")
    ohlcv = getter.get_feature(
        "spot_price_history",
        symbol="BTCUSDT",
        interval="1d",
        limit=1000,
    ).rename(columns={
        "spot_price_history__open":       "open",
        "spot_price_history__high":       "high",
        "spot_price_history__low":        "low",
        "spot_price_history__close":      "close",
        "spot_price_history__volume_usd": "volume",
    })

    dataset = df_wide.merge(ohlcv, on="date", how="inner")
    dataset = add_y_true(dataset, horizon=HORIZON)

    out_csv = out_dir / f"unbias_dataset_h{HORIZON}.csv"
    dataset.to_csv(out_csv, index=False)
    print(f"saved merged-> {out_csv}  shape={dataset.shape}")
    print(f"\ny_true distribution:\n{dataset['y_true'].value_counts(dropna=False)}")

    cm, metrics = build_confusion_matrix(dataset, threshold=50.0)
    print(f"\n=== confusion matrix (threshold=50, horizon={HORIZON}d, n={metrics['n']}) ===")
    print(cm)
    print(
        "\naccuracy={accuracy:.4f} | "
        "precision[LONG]={precision_long:.4f} recall[LONG]={recall_long:.4f} | "
        "precision[SHORT]={precision_short:.4f} recall[SHORT]={recall_short:.4f}".format(**metrics)
    )

    sweep = sweep_threshold(dataset, metric="accuracy")
    best = sweep.iloc[0]
    print(f"\n=== threshold sweep (top 5 by accuracy, horizon={HORIZON}d) ===")
    print(sweep.head(5).to_string(index=False))

    best_thr = float(best["thr"])
    cm_best, m_best = build_confusion_matrix(dataset, threshold=best_thr)
    print(f"\n=== best threshold = {best_thr} (accuracy={m_best['accuracy']:.4f}, n={m_best['n']}) ===")
    print(cm_best)
