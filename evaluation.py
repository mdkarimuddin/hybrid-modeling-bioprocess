"""
Evaluation and Visualization Module for Hybrid Modeling

Provides:
- Model evaluation metrics
- Visualization tools
- Comparison with mechanistic-only models
- Prediction analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import torch

# Try to import seaborn, but make it optional due to potential NumPy version conflicts
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except (ImportError, AttributeError):
    SEABORN_AVAILABLE = False
    # Create a dummy sns object if seaborn is not available
    class DummySeaborn:
        def set_style(self, *args, **kwargs):
            pass
        def set_palette(self, *args, **kwargs):
            pass
    sns = DummySeaborn()

# Try to import sklearn, but make it optional due to potential NumPy version conflicts
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define fallback functions if sklearn is not available
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

from hybrid_model import HybridModel, MechanisticModel


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    metrics : dict
        Dictionary of metrics
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN and Inf
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    metrics = {
        'mse': mean_squared_error(y_true_flat, y_pred_flat),
        'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        'mae': mean_absolute_error(y_true_flat, y_pred_flat),
        'r2': r2_score(y_true_flat, y_pred_flat),
        'mape': np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    }
    
    return metrics


def evaluate_model(model: HybridModel, test_data: np.ndarray,
                  t_span: np.ndarray, device: str = 'cpu') -> Dict:
    """
    Evaluate hybrid model on test data.
    
    Parameters:
    -----------
    model : HybridModel
        Trained hybrid model
    test_data : np.ndarray
        Test data of shape (n_samples, 3) for [X, S, P]
    t_span : np.ndarray
        Time points
    device : str
        Device for computation
    
    Returns:
    --------
    results : dict
        Evaluation results
    """
    model.eval()
    
    predictions = []
    mechanistic_predictions = []
    
    with torch.no_grad():
        for i in range(len(test_data) - 1):
            # Initial condition
            y0 = test_data[i]
            
            # Hybrid prediction
            pred_hybrid = model.predict_trajectory(
                t_span[i:i+2], y0, device=device
            )
            predictions.append(pred_hybrid[-1])
            
            # Mechanistic-only prediction
            pred_mech = model.mechanistic.solve_ode(
                t_span[i:i+2], y0
            )
            mechanistic_predictions.append(pred_mech[-1])
    
    predictions = np.array(predictions)
    mechanistic_predictions = np.array(mechanistic_predictions)
    targets = test_data[1:]
    
    # Compute metrics
    hybrid_metrics = compute_metrics(targets, predictions)
    mechanistic_metrics = compute_metrics(targets, mechanistic_predictions)
    
    results = {
        'hybrid_metrics': hybrid_metrics,
        'mechanistic_metrics': mechanistic_metrics,
        'predictions': predictions,
        'mechanistic_predictions': mechanistic_predictions,
        'targets': targets
    }
    
    return results


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history.
    
    Parameters:
    -----------
    history : dict
        Training history dictionary
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Data vs Physics loss
    axes[0, 1].plot(history['train_data_loss'], label='Train Data Loss', linewidth=2)
    axes[0, 1].plot(history['train_physics_loss'], label='Train Physics Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training: Data vs Physics Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation data vs physics loss
    axes[1, 0].plot(history['val_data_loss'], label='Val Data Loss', linewidth=2)
    axes[1, 0].plot(history['val_physics_loss'], label='Val Physics Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Validation: Data vs Physics Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(history['learning_rate'], label='Learning Rate', linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions(results: Dict, save_path: Optional[str] = None,
                    n_samples: int = 5):
    """
    Plot model predictions vs true values.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model
    save_path : str, optional
        Path to save figure
    n_samples : int
        Number of sample trajectories to plot
    """
    targets = results['targets']
    predictions = results['predictions']
    mechanistic_predictions = results['mechanistic_predictions']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    variables = ['Biomass (X)', 'Substrate (S)', 'Product (P)']
    colors = ['blue', 'green', 'red']
    
    for var_idx, (var_name, color) in enumerate(zip(variables, colors)):
        ax = axes[var_idx]
        
        # Plot sample trajectories
        n_plot = min(n_samples, len(targets))
        indices = np.linspace(0, len(targets) - 1, n_plot, dtype=int)
        
        for idx in indices:
            ax.plot([idx, idx+1], [targets[idx, var_idx], targets[idx+1, var_idx] if idx+1 < len(targets) else targets[idx, var_idx]],
                   'o-', color='black', alpha=0.3, label='True' if idx == indices[0] else '')
            ax.plot([idx, idx+1], [predictions[idx, var_idx], predictions[idx+1, var_idx] if idx+1 < len(predictions) else predictions[idx, var_idx]],
                   's-', color=color, alpha=0.5, label='Hybrid' if idx == indices[0] else '')
            ax.plot([idx, idx+1], [mechanistic_predictions[idx, var_idx], mechanistic_predictions[idx+1, var_idx] if idx+1 < len(mechanistic_predictions) else mechanistic_predictions[idx, var_idx]],
                   '^-', color='gray', alpha=0.5, label='Mechanistic' if idx == indices[0] else '')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(var_name)
        ax.set_title(f'{var_name} Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_scatter(results: Dict, save_path: Optional[str] = None):
    """
    Plot scatter plots: predicted vs true values.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model
    save_path : str, optional
        Path to save figure
    """
    targets = results['targets']
    predictions = results['predictions']
    mechanistic_predictions = results['mechanistic_predictions']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    variables = ['Biomass (X)', 'Substrate (S)', 'Product (P)']
    
    for var_idx, var_name in enumerate(variables):
        ax = axes[var_idx]
        
        # Hybrid predictions
        ax.scatter(targets[:, var_idx], predictions[:, var_idx],
                  alpha=0.5, label='Hybrid', s=20)
        
        # Mechanistic predictions
        ax.scatter(targets[:, var_idx], mechanistic_predictions[:, var_idx],
                  alpha=0.5, label='Mechanistic', s=20, marker='^')
        
        # Perfect prediction line
        min_val = min(targets[:, var_idx].min(), predictions[:, var_idx].min())
        max_val = max(targets[:, var_idx].max(), predictions[:, var_idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel(f'True {var_name}')
        ax.set_ylabel(f'Predicted {var_name}')
        ax.set_title(f'{var_name}: Predicted vs True')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Plot comparison of metrics between hybrid and mechanistic models.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model
    save_path : str, optional
        Path to save figure
    """
    hybrid_metrics = results['hybrid_metrics']
    mechanistic_metrics = results['mechanistic_metrics']
    
    metrics_names = ['RMSE', 'MAE', 'R²', 'MAPE']
    hybrid_values = [
        hybrid_metrics['rmse'],
        hybrid_metrics['mae'],
        hybrid_metrics['r2'],
        hybrid_metrics['mape']
    ]
    mechanistic_values = [
        mechanistic_metrics['rmse'],
        mechanistic_metrics['mae'],
        mechanistic_metrics['r2'],
        mechanistic_metrics['mape']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, hybrid_values, width, label='Hybrid Model', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, mechanistic_values, width, label='Mechanistic Model', color='gray', alpha=0.7)
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_evaluation_report(results: Dict):
    """
    Print detailed evaluation report.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model
    """
    hybrid_metrics = results['hybrid_metrics']
    mechanistic_metrics = results['mechanistic_metrics']
    
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print("\nHYBRID MODEL METRICS:")
    print(f"  RMSE: {hybrid_metrics['rmse']:.6f}")
    print(f"  MAE:  {hybrid_metrics['mae']:.6f}")
    print(f"  R²:   {hybrid_metrics['r2']:.6f}")
    print(f"  MAPE: {hybrid_metrics['mape']:.2f}%")
    
    print("\nMECHANISTIC MODEL METRICS:")
    print(f"  RMSE: {mechanistic_metrics['rmse']:.6f}")
    print(f"  MAE:  {mechanistic_metrics['mae']:.6f}")
    print(f"  R²:   {mechanistic_metrics['r2']:.6f}")
    print(f"  MAPE: {mechanistic_metrics['mape']:.2f}%")
    
    print("\nIMPROVEMENT:")
    rmse_improvement = ((mechanistic_metrics['rmse'] - hybrid_metrics['rmse']) / 
                       mechanistic_metrics['rmse'] * 100)
    r2_improvement = ((hybrid_metrics['r2'] - mechanistic_metrics['r2']) / 
                     (1 - mechanistic_metrics['r2'] + 1e-8) * 100)
    
    print(f"  RMSE Improvement: {rmse_improvement:.2f}%")
    print(f"  R² Improvement: {r2_improvement:.2f}%")
    print("=" * 60)

