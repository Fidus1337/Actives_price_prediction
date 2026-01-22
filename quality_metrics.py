import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd

def oos_proba_logreg(df: pd.DataFrame, features: list[str], target: str = "y_up_1d", n_splits: int = 5):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date", kind="stable").reset_index(drop=True)

    X = d[features].copy()
    y = pd.to_numeric(d[target], errors="coerce")

    m = y.notna() & X.notna().any(axis=1)
    X, y = X.loc[m].reset_index(drop=True), y.loc[m].astype(int).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    proba_oos = np.full(len(X), np.nan)
    for train_idx, test_idx in tscv.split(X):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba_oos[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]

    out = pd.DataFrame({"y": y, "p": proba_oos}).dropna()
    return out["y"].values, out["p"].values


def plot_roc(y, p, title="ROC"):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.show()
    return auc


def plot_metrics_vs_threshold(y_true, y_proba, title="Metrics vs threshold (OOF)", 
                               thresholds=None, figsize=(10, 6), show_best_f1=True):
    """
    Строит график зависимости метрик (Accuracy, Precision, Recall, F1) от порога.
    
    Parameters:
    -----------
    y_true : array-like
        Истинные метки классов (0 или 1)
    y_proba : array-like
        Вероятности положительного класса
    title : str
        Заголовок графика
    thresholds : array-like, optional
        Массив порогов для анализа. По умолчанию np.arange(0.01, 1.0, 0.01)
    figsize : tuple
        Размер графика
    show_best_f1 : bool
        Показывать вертикальную линию на лучшем F1
        
    Returns:
    --------
    dict : Словарь с лучшими значениями метрик и соответствующими порогами
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)
        
        # Precision (с защитой от деления на 0)
        if y_pred.sum() > 0:
            prec = precision_score(y_true, y_pred, zero_division=0)
        else:
            prec = 0.0
        precisions.append(prec)
        
        # Recall
        rec = recall_score(y_true, y_pred, zero_division=0)
        recalls.append(rec)
        
        # F1
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    # Находим лучшие значения
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thr = thresholds[best_f1_idx]
    best_f1_val = f1_scores[best_f1_idx]
    
    best_acc_idx = np.argmax(accuracies)
    best_acc_thr = thresholds[best_acc_idx]
    best_acc_val = accuracies[best_acc_idx]
    
    best_prec_idx = np.argmax(precisions)
    best_prec_thr = thresholds[best_prec_idx]
    best_prec_val = precisions[best_prec_idx]
    
    # Построение графика
    plt.figure(figsize=figsize)
    
    plt.plot(thresholds, accuracies, label="Accuracy", color="blue", linewidth=2)
    plt.plot(thresholds, precisions, label="Precision", color="orange", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall", color="green", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1", color="red", linewidth=2)
    
    if show_best_f1:
        plt.axvline(x=best_f1_thr, color="red", linestyle="--", alpha=0.7, 
                   label=f"Best F1 @ {best_f1_thr:.2f}")
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    # Возвращаем результаты
    results = {
        "best_f1": {"threshold": best_f1_thr, "value": best_f1_val},
        "best_accuracy": {"threshold": best_acc_thr, "value": best_acc_val},
        "best_precision": {"threshold": best_prec_thr, "value": best_prec_val},
        "metrics_at_best_f1": {
            "accuracy": accuracies[best_f1_idx],
            "precision": precisions[best_f1_idx],
            "recall": recalls[best_f1_idx],
            "f1": best_f1_val
        }
    }
    
    return results


def print_threshold_analysis(results: dict, model_name: str = "Model"):
    """Печатает результаты анализа порогов в читаемом формате."""
    print(f"\n{'='*50}")
    print(f"Threshold Analysis: {model_name}")
    print(f"{'='*50}")
    print(f"Best F1 Score: {results['best_f1']['value']:.4f} @ threshold = {results['best_f1']['threshold']:.2f}")
    print(f"Best Accuracy: {results['best_accuracy']['value']:.4f} @ threshold = {results['best_accuracy']['threshold']:.2f}")
    print(f"Best Precision: {results['best_precision']['value']:.4f} @ threshold = {results['best_precision']['threshold']:.2f}")
    print(f"\nMetrics at Best F1 threshold ({results['best_f1']['threshold']:.2f}):")
    for metric, value in results['metrics_at_best_f1'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    print(f"{'='*50}")