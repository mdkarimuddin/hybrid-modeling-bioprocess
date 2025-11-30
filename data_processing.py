"""
Data Processing Module for Hybrid Modeling Pipeline

Handles:
- Data loading and preprocessing
- Synthetic data generation for bioprocess
- Feature engineering
- Data normalization
- Train/validation/test splitting
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')


class BioprocessDataset(Dataset):
    """
    Dataset class for bioprocess time series data.
    """
    
    def __init__(self, data: np.ndarray, sequence_length: int = 10, 
                 normalize: bool = True, scaler: Optional[Dict] = None):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data of shape (n_samples, n_features)
            Features: [X (biomass), S (substrate), P (product), ...]
        sequence_length : int
            Length of input sequences for LSTM
        normalize : bool
            Whether to normalize data
        scaler : dict, optional
            Pre-computed normalization parameters (mean, std)
        """
        self.data = data
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Normalize data
        if normalize:
            if scaler is None:
                self.mean = np.mean(data, axis=0, keepdims=True)
                self.std = np.std(data, axis=0, keepdims=True) + 1e-8
                self.scaler = {'mean': self.mean, 'std': self.std}
            else:
                self.mean = scaler['mean']
                self.std = scaler['std']
                self.scaler = scaler
            
            self.data_normalized = (data - self.mean) / self.std
        else:
            self.data_normalized = data
            self.scaler = None
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.sequence_length)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sequence and target.
        
        Returns:
        --------
        sequence : torch.Tensor
            Input sequence of shape (sequence_length, n_features)
        target : torch.Tensor
            Target (next time step) of shape (n_features,)
        """
        sequence = self.data_normalized[idx:idx + self.sequence_length]
        target = self.data_normalized[idx + self.sequence_length]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


def generate_synthetic_bioprocess_data(
    n_experiments: int = 10,
    n_timepoints: int = 50,
    t_max: float = 100.0,
    noise_level: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic bioprocess data using mechanistic model.
    
    Parameters:
    -----------
    n_experiments : int
        Number of different experimental conditions
    n_timepoints : int
        Number of time points per experiment
    t_max : float
        Maximum time (hours)
    noise_level : float
        Relative noise level (std as fraction of signal)
    seed : int
        Random seed
    
    Returns:
    --------
    data : np.ndarray
        Data array of shape (n_experiments * n_timepoints, 3)
        Columns: [X (biomass), S (substrate), P (product)]
    time : np.ndarray
        Time points
    params : dict
        Parameters used for generation
    """
    np.random.seed(seed)
    
    # Define parameter ranges (realistic for mammalian cell culture)
    param_ranges = {
        'mu_max': (0.3, 0.7),  # 1/h
        'Ks': (0.05, 0.2),      # g/L
        'Yxs': (0.4, 0.6),      # g/g
        'Yps': (0.2, 0.4),      # g/g
        'qp_max': (0.05, 0.15)  # g/g/h
    }
    
    # Initial condition ranges
    X0_range = (0.1, 0.5)  # 10^6 cells/mL
    S0_range = (5.0, 15.0)  # g/L
    P0_range = (0.0, 0.1)   # g/L
    
    all_data = []
    all_times = []
    all_params = []
    
    for exp in range(n_experiments):
        # Sample parameters
        params = {
            'mu_max': np.random.uniform(*param_ranges['mu_max']),
            'Ks': np.random.uniform(*param_ranges['Ks']),
            'Yxs': np.random.uniform(*param_ranges['Yxs']),
            'Yps': np.random.uniform(*param_ranges['Yps']),
            'qp_max': np.random.uniform(*param_ranges['qp_max'])
        }
        
        # Sample initial conditions
        y0 = np.array([
            np.random.uniform(*X0_range),
            np.random.uniform(*S0_range),
            np.random.uniform(*P0_range)
        ])
        
        # Time points
        t = np.linspace(0, t_max, n_timepoints)
        
        # Solve ODE
        def ode_system(y, t, params):
            X, S, P = y
            mu = params['mu_max'] * S / (params['Ks'] + S + 1e-8)
            qp = params['qp_max'] * (S / (params['Ks'] + S + 1e-8))
            
            dX_dt = mu * X
            dS_dt = -(1.0 / params['Yxs']) * mu * X
            dP_dt = qp * X
            
            return [dX_dt, dS_dt, dP_dt]
        
        solution = odeint(ode_system, y0, t, args=(params,))
        
        # Add noise
        noise = np.random.normal(0, noise_level, solution.shape)
        solution_noisy = solution * (1 + noise)
        solution_noisy = np.maximum(solution_noisy, 0)  # Ensure non-negative
        
        all_data.append(solution_noisy)
        all_times.append(t)
        all_params.append(params)
    
    # Concatenate all experiments
    data = np.vstack(all_data)
    time = np.hstack(all_times)
    
    # Create parameter summary
    params_summary = {
        'n_experiments': n_experiments,
        'param_ranges': param_ranges,
        'individual_params': all_params
    }
    
    return data, time, params_summary


def prepare_sequences(data: np.ndarray, sequence_length: int = 10,
                      prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data of shape (n_samples, n_features)
    sequence_length : int
        Length of input sequences
    prediction_horizon : int
        How many steps ahead to predict
    
    Returns:
    --------
    X : np.ndarray
        Input sequences of shape (n_sequences, sequence_length, n_features)
    y : np.ndarray
        Target sequences of shape (n_sequences, n_features)
    """
    n_samples, n_features = data.shape
    
    X = []
    y = []
    
    for i in range(n_samples - sequence_length - prediction_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length + prediction_horizon - 1])
    
    return np.array(X), np.array(y)


def split_data(data: np.ndarray, train_ratio: float = 0.7,
               val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    train_ratio : float
        Fraction for training
    val_ratio : float
        Fraction for validation
    
    Returns:
    --------
    train_data : np.ndarray
    val_data : np.ndarray
    test_data : np.ndarray
    """
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    return train_data, val_data, test_data


def create_dataloaders(train_data: np.ndarray, val_data: np.ndarray,
                      test_data: np.ndarray, sequence_length: int = 10,
                      batch_size: int = 32, normalize: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Parameters:
    -----------
    train_data : np.ndarray
        Training data
    val_data : np.ndarray
        Validation data
    test_data : np.ndarray
        Test data
    sequence_length : int
        Sequence length for LSTM
    batch_size : int
        Batch size
    normalize : bool
        Whether to normalize
    
    Returns:
    --------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    scaler : dict
        Normalization parameters
    """
    # Create datasets
    train_dataset = BioprocessDataset(train_data, sequence_length, normalize=True)
    scaler = train_dataset.scaler
    
    val_dataset = BioprocessDataset(val_data, sequence_length, normalize=True, scaler=scaler)
    test_dataset = BioprocessDataset(test_data, sequence_length, normalize=True, scaler=scaler)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler


def load_real_data(filepath: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load real bioprocess data from file.
    
    Parameters:
    -----------
    filepath : str
        Path to data file (CSV, Excel, etc.)
    columns : list, optional
        Column names to use
    
    Returns:
    --------
    data : pd.DataFrame
        Loaded data
    """
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    if columns:
        data = data[columns]
    
    return data

