"""
Training Pipeline for Hybrid Modeling

Implements:
- Training loop with physics-informed loss
- Model validation
- Early stopping
- Learning rate scheduling
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import json
import time

from hybrid_model import HybridModel, PhysicsInformedLoss


class Trainer:
    """
    Trainer class for hybrid model.
    """
    
    def __init__(self, model: HybridModel, device: str = 'cpu',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        model : HybridModel
            Hybrid model to train
        device : str
            Device ('cpu' or 'cuda')
        learning_rate : float
            Initial learning rate
        weight_decay : float
            L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function
        self.criterion = PhysicsInformedLoss(
            lambda_physics=1.0,
            lambda_data=1.0
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'val_data_loss': [],
            'val_physics_loss': [],
            'learning_rate': []
        }
    
    def compute_ode_residual(self, states: torch.Tensor, predictions: torch.Tensor,
                            dt: float = 1.0) -> torch.Tensor:
        """
        Compute ODE residual for physics loss.
        
        For dX/dt = mu * X, residual = dX/dt - mu * X
        
        Parameters:
        -----------
        states : torch.Tensor
            Current states [X, S, P]
        predictions : torch.Tensor
            Predicted growth rates or derivatives
        dt : float
            Time step
        
        Returns:
        --------
        residual : torch.Tensor
            ODE residual
        """
        batch_size, seq_length, n_features = states.shape
        
        # Approximate derivatives using finite differences
        dX_dt = (states[:, 1:, 0] - states[:, :-1, 0]) / dt
        dS_dt = (states[:, 1:, 1] - states[:, :-1, 1]) / dt
        
        # Predicted growth rates
        mu_pred = predictions[:, :-1, 0]
        
        # ODE residuals
        residual_X = dX_dt - mu_pred * states[:, :-1, 0]
        residual_S = dS_dt + (1.0 / self.model.mechanistic.Yxs) * mu_pred * states[:, :-1, 0]
        
        # Combine residuals
        residual = torch.cat([residual_X.unsqueeze(-1), residual_S.unsqueeze(-1)], dim=-1)
        
        return residual
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        
        Returns:
        --------
        metrics : dict
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get predictions (growth rates)
            predictions = self.model(sequences, None)
            
            # For simplicity, compare with mechanistic prediction
            # In practice, you'd compare with actual growth rates from data
            mechanistic_pred = torch.zeros_like(predictions)
            for b in range(sequences.shape[0]):
                for s in range(sequences.shape[1]):
                    S = sequences[b, s, 1]
                    mechanistic_pred[b, s, 0] = self.model.mechanistic.growth_rate(S)
            
            # Compute ODE residual
            dt = 1.0  # Assuming unit time steps
            ode_residuals = self.compute_ode_residual(sequences, predictions, dt)
            
            # Compute loss
            # Use last prediction vs target
            pred_last = predictions[:, -1, :]
            target_last = mechanistic_pred[:, -1, :]  # Simplified target
            
            loss, data_loss, physics_loss = self.criterion(
                pred_last, target_last, ode_residuals
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'data_loss': total_data_loss / n_batches,
            'physics_loss': total_physics_loss / n_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
        
        Returns:
        --------
        metrics : dict
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(sequences, None)
                
                # Compute mechanistic predictions
                mechanistic_pred = torch.zeros_like(predictions)
                for b in range(sequences.shape[0]):
                    for s in range(sequences.shape[1]):
                        S = sequences[b, s, 1]
                        mechanistic_pred[b, s, 0] = self.model.mechanistic.growth_rate(S)
                
                # Compute ODE residual
                dt = 1.0
                ode_residuals = self.compute_ode_residual(sequences, predictions, dt)
                
                # Compute loss
                pred_last = predictions[:, -1, :]
                target_last = mechanistic_pred[:, -1, :]
                
                loss, data_loss, physics_loss = self.criterion(
                    pred_last, target_last, ode_residuals
                )
                
                total_loss += loss.item()
                total_data_loss += data_loss.item()
                total_physics_loss += physics_loss.item()
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'data_loss': total_data_loss / n_batches,
            'physics_loss': total_physics_loss / n_batches
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             n_epochs: int = 100, early_stopping_patience: int = 20,
             checkpoint_dir: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        Full training loop.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        n_epochs : int
            Number of training epochs
        early_stopping_patience : int
            Early stopping patience
        checkpoint_dir : str, optional
            Directory to save checkpoints
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        history : dict
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_data_loss'].append(train_metrics['data_loss'])
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['val_data_loss'].append(val_metrics['data_loss'])
            self.history['val_physics_loss'].append(val_metrics['physics_loss'])
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.6f} "
                      f"(Data: {train_metrics['data_loss']:.6f}, "
                      f"Physics: {train_metrics['physics_loss']:.6f})")
                print(f"  Val Loss: {val_metrics['loss']:.6f} "
                      f"(Data: {val_metrics['data_loss']:.6f}, "
                      f"Physics: {val_metrics['physics_loss']:.6f})")
                print(f"  LR: {current_lr:.6f}")
                print()
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_model.pt'),
                        epoch, val_metrics
                    )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict):
        """
        Save model checkpoint.
        
        Parameters:
        -----------
        filepath : str
            Path to save checkpoint
        epoch : int
            Current epoch
        metrics : dict
            Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Parameters:
        -----------
        filepath : str
            Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch'], checkpoint['metrics']

