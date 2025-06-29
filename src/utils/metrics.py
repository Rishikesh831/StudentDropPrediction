"""
Metrics utility for student dropout prediction model evaluation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def compute_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute confusion matrix and related metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with confusion matrix and metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }


def compute_roc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Compute ROC curve and AUC score
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary with ROC metrics
    """
    try:
        auc_score = roc_auc_score(y_true, y_prob)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        return {
            'auc_score': auc_score,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    except ValueError as e:
        print(f"Error computing ROC metrics: {e}")
        return {
            'auc_score': 0.0,
            'fpr': np.array([]),
            'tpr': np.array([]),
            'thresholds': np.array([])
        }


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Compute comprehensive set of metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary with all metrics
    """
    # Basic metrics
    basic_metrics = compute_basic_metrics(y_true, y_pred)
    
    # Confusion matrix metrics
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    
    # ROC metrics if probabilities are provided
    roc_metrics = {}
    if y_prob is not None:
        roc_metrics = compute_roc_metrics(y_true, y_prob)
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **cm_metrics, **roc_metrics}
    
    return all_metrics


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Dictionary of metrics
    """
    print("=" * 50)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 50)
    
    # Basic metrics
    print("\n📊 BASIC METRICS:")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:  {metrics.get('f1_score', 0):.4f}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        print("\n🔍 CONFUSION MATRIX:")
        cm = metrics['confusion_matrix']
        print("           Predicted")
        print("           0    1")
        print(f"Actual 0 [{cm[0,0]:3d} {cm[0,1]:3d}]")
        print(f"      1 [{cm[1,0]:3d} {cm[1,1]:3d}]")
    
    # Additional metrics
    print("\n📈 ADDITIONAL METRICS:")
    print(f"Specificity:        {metrics.get('specificity', 0):.4f}")
    print(f"Sensitivity:        {metrics.get('sensitivity', 0):.4f}")
    print(f"False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}")
    print(f"False Negative Rate: {metrics.get('false_negative_rate', 0):.4f}")
    
    # ROC metrics
    if 'auc_score' in metrics:
        print(f"AUC Score:          {metrics.get('auc_score', 0):.4f}")
    
    print("=" * 50)


def plot_confusion_matrix(metrics: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        metrics: Dictionary of metrics
        save_path: Optional path to save the plot
    """
    if 'confusion_matrix' not in metrics:
        print("No confusion matrix found in metrics")
        return
    
    cm = metrics['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Dropout', 'Dropout'],
                yticklabels=['No Dropout', 'Dropout'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(metrics: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot ROC curve
    
    Args:
        metrics: Dictionary of metrics
        save_path: Optional path to save the plot
    """
    if 'fpr' not in metrics or 'tpr' not in metrics:
        print("No ROC curve data found in metrics")
        return
    
    fpr = metrics['fpr']
    tpr = metrics['tpr']
    auc_score = metrics.get('auc_score', 0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None, 
                             plot_results: bool = True, save_plots: bool = False, 
                             plot_prefix: str = "model_evaluation") -> Dict[str, Any]:
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        plot_results: Whether to plot results
        save_plots: Whether to save plots
        plot_prefix: Prefix for saved plot files
        
    Returns:
        Dictionary with all metrics
    """
    # Compute metrics
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Plot results if requested
    if plot_results:
        if save_plots:
            cm_path = f"{plot_prefix}_confusion_matrix.png"
            roc_path = f"{plot_prefix}_roc_curve.png"
        else:
            cm_path = None
            roc_path = None
        
        plot_confusion_matrix(metrics, cm_path)
        
        if y_prob is not None:
            plot_roc_curve(metrics, roc_path)
    
    return metrics


def calculate_prediction_confidence(y_prob: np.ndarray) -> np.ndarray:
    """
    Calculate prediction confidence based on probability distance from 0.5
    
    Args:
        y_prob: Predicted probabilities
        
    Returns:
        Confidence scores
    """
    return np.abs(y_prob - 0.5) * 2  # Scale to [0, 1]


def get_high_confidence_predictions(y_prob: np.ndarray, confidence_threshold: float = 0.8) -> np.ndarray:
    """
    Get predictions with high confidence
    
    Args:
        y_prob: Predicted probabilities
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Boolean array indicating high confidence predictions
    """
    confidence = calculate_prediction_confidence(y_prob)
    return confidence >= confidence_threshold 