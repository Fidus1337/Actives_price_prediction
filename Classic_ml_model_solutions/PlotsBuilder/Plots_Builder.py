import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import os
import shutil
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd

# Папка для сохранения графиков рядом с Plots_Builder.py:
# Classic_ml_model_solutions/PlotsBuilder/plots
_PLOTS_BUILDER_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHICS_DIR = os.path.join(_PLOTS_BUILDER_DIR, "plots")
_MIGRATION_DONE = False


def _next_available_path(dst_file: Path) -> Path:
    """Return a free path by adding _migratedN suffix when needed."""
    if not dst_file.exists():
        return dst_file

    stem = dst_file.stem
    suffix = dst_file.suffix
    parent = dst_file.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_migrated{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _iter_legacy_plot_dirs(project_root: Path):
    """
    Yield possible legacy plot directories.

    Supported legacy locations:
    - <repo>/Models/plots
    - <repo>/Classic_ml_model_solutions/Created_models_to_use/plots
    - <repo>/Classic_ml_model_solutions/Created_models_to_use/<config>/plots
    """
    # Old relative path from the previous implementation.
    yield project_root / "Models" / "plots"
    yield project_root / "Classic_ml_model_solutions" / "Created_models_to_use" / "plots"

    created_models_root = project_root / "Classic_ml_model_solutions" / "Created_models_to_use"
    if created_models_root.exists():
        for cfg_dir in created_models_root.iterdir():
            if cfg_dir.is_dir():
                legacy_cfg_plots = cfg_dir / "plots"
                if legacy_cfg_plots.exists():
                    yield legacy_cfg_plots


def migrate_legacy_plots():
    """
    Move legacy plots to GRAPHICS_DIR once.

    Files are moved without overwriting existing files.
    """
    global _MIGRATION_DONE
    if _MIGRATION_DONE:
        return
    _MIGRATION_DONE = True

    destination_root = Path(GRAPHICS_DIR)
    destination_root.mkdir(parents=True, exist_ok=True)

    project_root = Path(_PLOTS_BUILDER_DIR).parent.parent
    moved_files = 0

    for legacy_dir in _iter_legacy_plot_dirs(project_root):
        legacy_dir = Path(legacy_dir)
        if not legacy_dir.exists() or not legacy_dir.is_dir():
            continue
        if legacy_dir.resolve() == destination_root.resolve():
            continue

        for src_file in legacy_dir.rglob("*"):
            if not src_file.is_file():
                continue

            rel = src_file.relative_to(legacy_dir)
            dst_file = destination_root / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file = _next_available_path(dst_file)
            shutil.move(str(src_file), str(dst_file))
            moved_files += 1

        # Cleanup empty legacy tree (best-effort)
        for candidate in sorted(legacy_dir.rglob("*"), reverse=True):
            if candidate.is_dir():
                try:
                    candidate.rmdir()
                except OSError:
                    pass
        try:
            legacy_dir.rmdir()
        except OSError:
            pass

    if moved_files:
        print(f"Migrated {moved_files} legacy plot files to: {destination_root}")


def ensure_graphics_dir(config_name: str = None) -> str:
    """
    Создает папку для графиков если её нет.
    
    Parameters:
    -----------
    config_name : str, optional
        Имя конфигурации для создания подпапки
        
    Returns:
    --------
    str : Путь к созданной директории
    """
    migrate_legacy_plots()

    if config_name:
        path = os.path.join(GRAPHICS_DIR, config_name)
    else:
        path = GRAPHICS_DIR
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path


def plot_roc(y, p, title="ROC", save_path=None, config_name: str = "BASE"):
    """
    Строит ROC-кривую и сохраняет в файл.
    
    Parameters:
    -----------
    y : array-like
        Истинные метки
    p : array-like
        Вероятности
    title : str
        Заголовок графика
    save_path : str, optional
        Путь для сохранения графика. Если None, сохраняет в папку graphics/config_name с автоименем.
    config_name : str
        Имя конфигурации для подпапки
        
    Returns:
    --------
    float : AUC score
    """
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", label="random", alpha=0.7)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Создаём директорию с учётом config_name
    graphics_path = ensure_graphics_dir(config_name)
    
    if save_path is None:
        # Генерируем имя файла из заголовка
        filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") + ".png"
        save_path = os.path.join(graphics_path, filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC saved to: {save_path}")
    plt.close()
    
    return auc


def plot_metrics_vs_threshold(y_true, y_proba, title="Metrics vs threshold (OOF)", 
                               thresholds=None, figsize=(10, 6), show_best_f1=True, 
                               config_name: str = "BASE"):
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
    config_name : str
        Имя конфигурации для подпапки
        
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
    
    # Создаём директорию с учётом config_name
    graphics_path = ensure_graphics_dir(config_name)
    
    # Генерируем безопасное имя файла
    filename = (title.replace(" ", "_")
                     .replace("(", "")
                     .replace(")", "")
                     .replace(",", "")
                     .replace("-", "_") + ".png")
    save_path = os.path.join(graphics_path, filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Metrics plot saved to: {save_path}")
    plt.close()
    
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

def plot_confusion_matrix(y_true, y_proba, threshold=0.5,
                          title="Confusion Matrix", config_name="BASE"):
    """
    Строит и сохраняет confusion matrix.

    Parameters:
    -----------
    y_true : array-like
        Истинные метки классов (0 или 1)
    y_proba : array-like
        Вероятности положительного класса
    threshold : float
        Порог классификации
    title : str
        Заголовок графика
    config_name : str
        Имя конфигурации для подпапки

    Returns:
    --------
    np.ndarray : Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Down (0)", "Up (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{title}\nThreshold = {threshold:.2f}")

    # Сохранение
    graphics_path = ensure_graphics_dir(config_name)
    filename = f"Confusion_Matrix_{config_name}_thr{threshold:.2f}.png"
    save_path = os.path.join(graphics_path, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

    return cm