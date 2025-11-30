"""
Example Usage Script for Hybrid Modeling Pipeline

This script demonstrates how to:
1. Generate synthetic bioprocess data
2. Train a hybrid model
3. Evaluate and compare with mechanistic-only model
4. Visualize results
"""

import numpy as np
import torch
import matplotlib
# Set non-interactive backend for batch jobs (Puhti)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

from hybrid_model import HybridModel
from data_processing import (
    generate_synthetic_bioprocess_data,
    create_dataloaders,
    split_data
)
from training import Trainer
from evaluation import (
    evaluate_model,
    plot_training_history,
    plot_predictions,
    plot_prediction_scatter,
    plot_metrics_comparison,
    print_evaluation_report
)


def main():
    """
    Main example workflow.
    """
    print("=" * 60)
    print("HYBRID MODELING PIPELINE - EXAMPLE USAGE")
    print("=" * 60)
    print()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # ===================================================================
    # STEP 1: Generate Synthetic Bioprocess Data
    # ===================================================================
    print("STEP 1: Generating synthetic bioprocess data...")
    data, time, params_summary = generate_synthetic_bioprocess_data(
        n_experiments=20,
        n_timepoints=50,
        t_max=100.0,
        noise_level=0.05,
        seed=42
    )
    
    print(f"  Generated {len(data)} data points")
    print(f"  Data shape: {data.shape}")
    print(f"  Features: [Biomass (X), Substrate (S), Product (P)]")
    print()
    
    # ===================================================================
    # STEP 2: Prepare Data
    # ===================================================================
    print("STEP 2: Preparing data for training...")
    
    # Split data
    train_data, val_data, test_data = split_data(
        data, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Create dataloaders
    sequence_length = 10
    batch_size = 32
    
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_data, val_data, test_data,
        sequence_length=sequence_length,
        batch_size=batch_size,
        normalize=True
    )
    
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # ===================================================================
    # STEP 3: Initialize Hybrid Model
    # ===================================================================
    print("STEP 3: Initializing hybrid model...")
    
    # Mechanistic parameters (typical for mammalian cell culture)
    mechanistic_params = {
        'mu_max': 0.5,   # 1/h
        'Ks': 0.1,       # g/L
        'Yxs': 0.5,      # g/g
        'Yps': 0.3,      # g/g
        'qp_max': 0.1    # g/g/h
    }
    
    # Initialize model
    model = HybridModel(
        mechanistic_params=mechanistic_params,
        ml_input_dim=3,      # X, S, P
        ml_hidden_dim=64,
        ml_num_layers=2,
        use_residual_learning=True
    )
    
    print(f"  Model architecture:")
    print(f"    - Mechanistic component: ODE-based bioprocess model")
    print(f"    - ML component: LSTM with {64} hidden units, {2} layers")
    print(f"    - Residual learning: {True}")
    print()
    
    # ===================================================================
    # STEP 4: Train Model
    # ===================================================================
    print("STEP 4: Training hybrid model...")
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        early_stopping_patience=20,
        checkpoint_dir=str(checkpoint_dir),
        verbose=True
    )
    
    print()
    
    # ===================================================================
    # STEP 5: Visualize Training History
    # ===================================================================
    print("STEP 5: Visualizing training history...")
    plot_training_history(
        history,
        save_path=str(output_dir / "training_history.png")
    )
    print()
    
    # ===================================================================
    # STEP 6: Evaluate Model
    # ===================================================================
    print("STEP 6: Evaluating model on test set...")
    
    # Prepare test data for evaluation
    # For simplicity, use first part of test data
    test_eval_data = test_data[:100]
    t_span_eval = np.linspace(0, 10, len(test_eval_data))
    
    results = evaluate_model(
        model=model,
        test_data=test_eval_data,
        t_span=t_span_eval,
        device=device
    )
    
    # Print evaluation report
    print_evaluation_report(results)
    print()
    
    # ===================================================================
    # STEP 7: Visualize Results
    # ===================================================================
    print("STEP 7: Creating visualization plots...")
    
    # Plot predictions
    plot_predictions(
        results,
        save_path=str(output_dir / "predictions.png"),
        n_samples=5
    )
    
    # Plot scatter plots
    plot_prediction_scatter(
        results,
        save_path=str(output_dir / "prediction_scatter.png")
    )
    
    # Plot metrics comparison
    plot_metrics_comparison(
        results,
        save_path=str(output_dir / "metrics_comparison.png")
    )
    
    print()
    
    # ===================================================================
    # STEP 8: Save Model and Results
    # ===================================================================
    print("STEP 8: Saving model and results...")
    
    # Save final model
    model_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'mechanistic_params': mechanistic_params,
        'scaler': scaler,
        'history': history
    }, model_path)
    
    print(f"  Model saved to: {model_path}")
    print(f"  All outputs saved to: {output_dir}")
    print()
    
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review the generated plots in the 'outputs' directory")
    print("  2. Experiment with different hyperparameters")
    print("  3. Try with your own bioprocess data")
    print("  4. Extend the model for additional features (pH, temperature, etc.)")


if __name__ == "__main__":
    main()

