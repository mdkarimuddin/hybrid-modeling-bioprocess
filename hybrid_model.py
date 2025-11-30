"""
Hybrid Modeling Pipeline for Bioprocess Optimization
Combines mechanistic ODE models with machine learning (LSTM/neural networks)

This module implements a hybrid modeling approach similar to Yokogawa Insilico's
digital twin technology, combining first-principles equations with AI for
cell culture process prediction and optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MechanisticModel:
    """
    Mechanistic component: First-principles ODE model for cell culture bioprocess.
    
    Implements mass balance equations for:
    - X: Biomass concentration (cells/mL)
    - S: Substrate concentration (g/L)
    - P: Product concentration (g/L)
    """
    
    def __init__(self, params: Dict[str, float]):
        """
        Initialize mechanistic model with kinetic parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing kinetic parameters:
            - mu_max: Maximum specific growth rate (1/h)
            - Ks: Substrate saturation constant (g/L)
            - Yxs: Biomass yield on substrate (g/g)
            - Yps: Product yield on substrate (g/g)
            - qp_max: Maximum specific product formation rate (g/g/h)
        """
        self.params = params
        self.mu_max = params.get('mu_max', 0.5)
        self.Ks = params.get('Ks', 0.1)
        self.Yxs = params.get('Yxs', 0.5)
        self.Yps = params.get('Yps', 0.3)
        self.qp_max = params.get('qp_max', 0.1)
    
    def growth_rate(self, S: torch.Tensor) -> torch.Tensor:
        """
        Monod kinetics for specific growth rate.
        
        mu(S) = mu_max * S / (Ks + S)
        """
        return self.mu_max * S / (self.Ks + S + 1e-8)
    
    def ode_system(self, t: float, y: np.ndarray, mu: float) -> np.ndarray:
        """
        ODE system for cell culture:
        dX/dt = mu * X
        dS/dt = -(1/Yxs) * mu * X
        dP/dt = qp * X
        
        Parameters:
        -----------
        t : float
            Time point
        y : np.ndarray
            State vector [X, S, P]
        mu : float
            Specific growth rate (can be ML-predicted)
        
        Returns:
        --------
        dydt : np.ndarray
            Derivatives [dX/dt, dS/dt, dP/dt]
        """
        X, S, P = y
        
        # Growth rate (Monod kinetics)
        mu_actual = self.mu_max * S / (self.Ks + S + 1e-8)
        
        # Use provided mu if given (from ML component)
        if mu is not None:
            mu_actual = mu
        
        # Product formation rate (simplified)
        qp = self.qp_max * (S / (self.Ks + S + 1e-8))
        
        # ODEs
        dX_dt = mu_actual * X
        dS_dt = -(1.0 / self.Yxs) * mu_actual * X
        dP_dt = qp * X
        
        return np.array([dX_dt, dS_dt, dP_dt])
    
    def solve_ode(self, t_span: np.ndarray, y0: np.ndarray, 
                  mu_ml: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve ODE system using scipy.integrate.odeint.
        
        Parameters:
        -----------
        t_span : np.ndarray
            Time points for integration
        y0 : np.ndarray
            Initial conditions [X0, S0, P0]
        mu_ml : np.ndarray, optional
            ML-predicted growth rates at each time point
        
        Returns:
        --------
        solution : np.ndarray
            Solution array of shape (n_times, 3) for [X, S, P]
        """
        def ode_wrapper(y, t):
            if mu_ml is not None:
                # Interpolate ML-predicted mu
                idx = np.clip(np.searchsorted(t_span, t, side='right') - 1, 0, len(mu_ml) - 1)
                mu_val = mu_ml[idx]
            else:
                mu_val = None
            return self.ode_system(t, y, mu_val)
        
        solution = odeint(ode_wrapper, y0, t_span)
        return solution


class MLComponent(nn.Module):
    """
    Machine Learning component: LSTM network for learning complex kinetics.
    
    Learns:
    - Complex growth rate dependencies (beyond Monod)
    - Metabolic interactions
    - Environmental effects (pH, temperature, etc.)
    - Residual corrections to mechanistic predictions
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1):
        """
        Initialize LSTM network.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (e.g., S, X, P, pH, T)
        hidden_dim : int
            LSTM hidden dimension
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Output dimension (e.g., growth rate correction or complex mu)
        dropout : float
            Dropout rate
        """
        super(MLComponent, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_dim)
        
        Returns:
        --------
        output : torch.Tensor
            Output tensor of shape (batch_size, seq_length, output_dim)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Apply fully connected layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class HybridModel(nn.Module):
    """
    Hybrid Model: Combines mechanistic ODEs with ML component.
    
    Architecture:
    1. Mechanistic component provides base predictions (respects physics)
    2. ML component learns complex patterns and corrections
    3. Combined prediction = mechanistic + ML correction
    """
    
    def __init__(self, mechanistic_params: Dict[str, float],
                 ml_input_dim: int = 5, ml_hidden_dim: int = 64,
                 ml_num_layers: int = 2, use_residual_learning: bool = True):
        """
        Initialize hybrid model.
        
        Parameters:
        -----------
        mechanistic_params : dict
            Parameters for mechanistic model
        ml_input_dim : int
            Input dimension for ML component
        ml_hidden_dim : int
            Hidden dimension for ML component
        ml_num_layers : int
            Number of layers in ML component
        use_residual_learning : bool
            If True, ML learns residual corrections to mechanistic model
            If False, ML directly predicts growth rates
        """
        super(HybridModel, self).__init__()
        
        self.mechanistic = MechanisticModel(mechanistic_params)
        self.ml_component = MLComponent(
            input_dim=ml_input_dim,
            hidden_dim=ml_hidden_dim,
            num_layers=ml_num_layers,
            output_dim=1,
            dropout=0.1
        )
        self.use_residual_learning = use_residual_learning
        
    def forward(self, states: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Hybrid prediction.
        
        Parameters:
        -----------
        states : torch.Tensor
            State tensor of shape (batch_size, seq_length, state_dim)
            Contains [X, S, P, ...] at each time step
        t : torch.Tensor
            Time tensor of shape (batch_size, seq_length, 1)
        
        Returns:
        --------
        predictions : torch.Tensor
            Predicted derivatives or corrections
        """
        # Extract features for ML (X, S, P, and optionally pH, T, etc.)
        ml_input = states  # Shape: (batch, seq, features)
        
        # ML prediction
        ml_output = self.ml_component(ml_input)  # Shape: (batch, seq, 1)
        
        # Mechanistic prediction
        batch_size, seq_length, _ = states.shape
        mechanistic_pred = torch.zeros_like(ml_output)
        
        for b in range(batch_size):
            for s in range(seq_length):
                X = states[b, s, 0]
                S = states[b, s, 1]
                P = states[b, s, 2] if states.shape[2] > 2 else torch.tensor(0.0)
                
                # Mechanistic growth rate
                mu_mech = self.mechanistic.growth_rate(S)
                
                if self.use_residual_learning:
                    # ML learns correction to mechanistic prediction
                    mu_hybrid = mu_mech + ml_output[b, s, 0]
                else:
                    # ML directly predicts growth rate
                    mu_hybrid = ml_output[b, s, 0]
                
                mechanistic_pred[b, s, 0] = mu_hybrid
        
        return mechanistic_pred
    
    def predict_trajectory(self, t_span: np.ndarray, y0: np.ndarray,
                          states_history: Optional[np.ndarray] = None,
                          device: str = 'cpu') -> np.ndarray:
        """
        Predict full trajectory using hybrid model.
        
        Parameters:
        -----------
        t_span : np.ndarray
            Time points
        y0 : np.ndarray
            Initial conditions [X0, S0, P0]
        states_history : np.ndarray, optional
            Historical states for ML input
        device : str
            Device for computation ('cpu' or 'cuda')
        
        Returns:
        --------
        trajectory : np.ndarray
            Predicted trajectory of shape (n_times, 3)
        """
        self.eval()
        
        # Prepare ML input
        if states_history is None:
            # Use mechanistic prediction as initial guess
            states_history = self.mechanistic.solve_ode(t_span, y0)
        
        # Convert to torch
        states_tensor = torch.FloatTensor(states_history).unsqueeze(0).to(device)
        t_tensor = torch.FloatTensor(t_span).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Get ML predictions
        with torch.no_grad():
            ml_corrections = self.forward(states_tensor, t_tensor)
            ml_corrections = ml_corrections.squeeze(0).cpu().numpy()
        
        # Solve ODE with ML-corrected growth rates
        mu_ml = ml_corrections.flatten()
        trajectory = self.mechanistic.solve_ode(t_span, y0, mu_ml=mu_ml)
        
        return trajectory


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function.
    
    Combines:
    1. Data fitting loss (MSE with observations)
    2. Physics loss (ODE residual)
    3. Boundary condition loss
    """
    
    def __init__(self, lambda_physics: float = 1.0, lambda_data: float = 1.0):
        """
        Initialize loss function.
        
        Parameters:
        -----------
        lambda_physics : float
            Weight for physics loss term
        lambda_data : float
            Weight for data fitting loss term
        """
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.lambda_data = lambda_data
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                ode_residuals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Model predictions
        targets : torch.Tensor
            Target values (observations)
        ode_residuals : torch.Tensor, optional
            ODE residual terms (physics violations)
        
        Returns:
        --------
        loss : torch.Tensor
            Combined loss value
        """
        # Data fitting loss
        data_loss = self.mse(predictions, targets)
        
        # Physics loss (ODE residual)
        if ode_residuals is not None:
            physics_loss = torch.mean(ode_residuals ** 2)
        else:
            physics_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combined loss
        total_loss = (self.lambda_data * data_loss + 
                     self.lambda_physics * physics_loss)
        
        return total_loss, data_loss, physics_loss

