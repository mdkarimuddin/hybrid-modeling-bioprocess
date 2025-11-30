# What Do These Scripts Do?

This document explains what happens when you run the hybrid modeling pipeline scripts.

## Overview: The Complete Workflow

The pipeline implements a **hybrid modeling approach** that combines:
- **Mechanistic models** (physics-based ODEs) 
- **Machine learning** (LSTM neural networks)

This is exactly what Yokogawa Insilico does for bioprocess digital twins!

---

## Script 1: `test_environment_puhti.sh` (Quick Test - ~2 minutes)

### What it does:
1. **Loads Python environment**
   - Loads `python-data` module on Puhti
   - Sets up Python 3 environment

2. **Checks all required packages**
   - Tests if NumPy, SciPy, Pandas are available
   - Tests if PyTorch (for deep learning) is available
   - Tests if Matplotlib, Seaborn (for plotting) are available
   - Tests if scikit-learn (for metrics) is available

3. **Tests pipeline imports**
   - Tries to import `hybrid_model.py` module
   - Tries to import `data_processing.py` module
   - Tries to import `training.py` module
   - Tries to import `evaluation.py` module

4. **Reports results**
   - ✅ If everything works: "ALL TESTS PASSED"
   - ❌ If something fails: Shows what's missing

### Why run this first?
- **Fast** (only 2 minutes)
- **Catches problems early** before running the full pipeline
- **Saves time** - no point running 1-hour job if environment is broken

### Expected output:
```
✅ NumPy - version 1.21.0
✅ SciPy - version 1.7.0
✅ PyTorch - version 1.10.0
✅ hybrid_model module imported successfully
✅ data_processing module imported successfully
...
✅ ALL TESTS PASSED - Environment is ready!
```

---

## Script 2: `run_hybrid_modeling_puhti.sh` (Full Pipeline - ~30-60 minutes)

### What it does step-by-step:

#### STEP 1: Environment Setup (1-2 minutes)
- Loads `python-data` module
- Checks Python version
- Verifies all packages are available
- Sets matplotlib to non-interactive mode (for batch jobs)

#### STEP 2: Generate Synthetic Bioprocess Data (2-3 minutes)
- Creates **20 different experimental conditions**
- Each experiment has **50 time points** over 100 hours
- Simulates cell culture process with:
  - **X (Biomass)**: Cell concentration over time
  - **S (Substrate)**: Nutrient (glucose) consumption
  - **P (Product)**: Protein/drug production
- Adds realistic noise (5%) to simulate real experimental data
- **Total**: ~1000 data points

**Example data generated:**
```
Time (h) | Biomass (X) | Substrate (S) | Product (P)
---------|-------------|---------------|-------------
0        | 0.2         | 10.0          | 0.0
10       | 0.5         | 8.5           | 0.1
20       | 1.2         | 6.0           | 0.3
...
```

#### STEP 3: Prepare Data for Training (1 minute)
- **Splits data**:
  - 70% training (to teach the model)
  - 15% validation (to tune parameters)
  - 15% test (to evaluate final performance)
- **Creates sequences** for LSTM:
  - Input: Last 10 time points
  - Output: Next time point
- **Normalizes data** (scales to 0-1 range for better training)

#### STEP 4: Initialize Hybrid Model (instant)
- **Mechanistic component**:
  - ODE system: `dX/dt = μ(S) * X` (cell growth)
  - Uses Monod kinetics: `μ(S) = μ_max * S / (Ks + S)`
  - Respects mass balance (physics!)
- **ML component**:
  - LSTM network with 64 hidden units, 2 layers
  - Learns complex patterns the mechanistic model misses
- **Hybrid integration**:
  - Combines both: `μ_hybrid = μ_mechanistic + ML_correction`

#### STEP 5: Train the Model (~20-40 minutes)
This is the main computation step!

**What happens:**
1. **Forward pass**: Model makes predictions
2. **Compute loss**: 
   - Data loss: How well predictions match observations
   - Physics loss: How well predictions respect ODE constraints
3. **Backward pass**: Update model parameters (learning)
4. **Repeat** for 100 epochs (or until early stopping)

**Training progress:**
```
Epoch 10/100
  Train Loss: 0.023456 (Data: 0.015234, Physics: 0.008222)
  Val Loss: 0.025123 (Data: 0.016789, Physics: 0.008334)
  LR: 0.001000

Epoch 20/100
  Train Loss: 0.012345 (Data: 0.008901, Physics: 0.003444)
  Val Loss: 0.013456 (Data: 0.009234, Physics: 0.004222)
  LR: 0.001000
...
```

**Early stopping**: If validation loss doesn't improve for 20 epochs, training stops (prevents overfitting)

#### STEP 6: Evaluate Model Performance (~2-3 minutes)
- Tests model on **unseen test data**
- Compares **Hybrid model** vs **Mechanistic-only model**
- Computes metrics:
  - **RMSE** (Root Mean Squared Error) - lower is better
  - **MAE** (Mean Absolute Error) - lower is better
  - **R²** (Coefficient of determination) - higher is better (max = 1.0)
  - **MAPE** (Mean Absolute Percentage Error) - lower is better

**Expected results:**
```
HYBRID MODEL METRICS:
  RMSE: 0.045123
  MAE:  0.032456
  R²:   0.923456
  MAPE: 5.23%

MECHANISTIC MODEL METRICS:
  RMSE: 0.078901
  MAE:  0.056789
  R²:   0.812345
  MAPE: 8.45%

IMPROVEMENT:
  RMSE Improvement: 42.8%
  R² Improvement: 59.2%
```

#### STEP 7: Generate Visualizations (~1-2 minutes)
Creates 4 plots:

1. **`training_history.png`**:
   - Training/validation loss curves
   - Data loss vs Physics loss
   - Learning rate schedule
   - Shows if model is learning well

2. **`predictions.png`**:
   - Time series predictions
   - Shows how well model predicts future values
   - Compares Hybrid vs Mechanistic vs True values

3. **`prediction_scatter.png`**:
   - Predicted vs True values
   - Points on diagonal = perfect predictions
   - Shows prediction accuracy

4. **`metrics_comparison.png`**:
   - Bar chart comparing Hybrid vs Mechanistic
   - Visual summary of performance improvement

#### STEP 8: Save Results (~30 seconds)
- Saves trained model: `outputs/final_model.pt`
- Saves best model: `outputs/checkpoints/best_model.pt`
- Saves training history (for later analysis)
- All plots saved to `outputs/` directory

---

## What You Get After Running

### Files Created:

```
outputs/
├── training_history.png          # Training curves
├── predictions.png                # Time series predictions
├── prediction_scatter.png        # Accuracy plots
├── metrics_comparison.png        # Model comparison
├── final_model.pt                  # Trained model (can reload later)
└── checkpoints/
    └── best_model.pt             # Best model during training
```

### Log Files:

```
hybrid_modeling_<JOB_ID>.out      # Standard output (all print statements)
hybrid_modeling_<JOB_ID>.err      # Error messages (if any)
```

---

## What This Demonstrates (For Your Interview)

### Technical Skills:
✅ **Hybrid Modeling**: Combining mechanistic + ML  
✅ **Physics-Informed ML**: Enforcing physical constraints  
✅ **Deep Learning**: LSTM for sequence learning  
✅ **Data Science Pipeline**: End-to-end workflow  
✅ **Production Code**: Error handling, logging, checkpointing  

### Alignment with Yokogawa Insilico:
✅ **Digital Twin Technology**: Exactly their approach!  
✅ **Bioprocess Optimization**: Cell culture modeling  
✅ **Customer-Facing**: Complete, documented pipeline  
✅ **Scientific Rigor**: Physics constraints + data validation  

---

## Next Steps After Running

### 1. Review Results
```bash
# Check if job completed
cat hybrid_modeling_<JOB_ID>.out

# View the plots
ls -lh outputs/*.png

# Check model performance
grep "IMPROVEMENT" hybrid_modeling_<JOB_ID>.out
```

### 2. Experiment with Parameters
Edit `example_usage.py` to try:
- Different number of experiments
- Different model architectures (more/less LSTM layers)
- Different training parameters (learning rate, batch size)
- Different mechanistic parameters (for your specific cell line)

### 3. Use Your Own Data
Replace synthetic data generation with your real bioprocess data:
```python
# In example_usage.py, replace:
data, time, params = generate_synthetic_bioprocess_data(...)

# With:
from data_processing import load_real_data
data = load_real_data('your_data.csv')
```

### 4. Extend the Model
Add more features:
- pH measurements
- Temperature
- Dissolved oxygen
- Other process variables

---

## Expected Timeline

| Step | Time | Description |
|------|------|-------------|
| Environment test | 1-2 min | Quick validation |
| Full pipeline | 30-60 min | Complete run |
| - Data generation | 2-3 min | Create synthetic data |
| - Training | 20-40 min | Main computation |
| - Evaluation | 2-3 min | Test performance |
| - Visualization | 1-2 min | Generate plots |

**Total**: ~30-60 minutes depending on system load

---

## Troubleshooting

### If test fails:
- Check which package is missing
- May need different Python module
- Check error log: `test_env_<JOB_ID>.err`

### If full pipeline fails:
- Check memory usage (may need more RAM)
- Check time limit (may need more time)
- Reduce data size or batch size
- Check error log: `hybrid_modeling_<JOB_ID>.err`

---

## Summary

**Test script**: Quick validation (2 min)  
**Full pipeline**: Complete hybrid modeling workflow (30-60 min)

**What you get**:
- Trained hybrid model
- Performance metrics
- Visualization plots
- Model checkpoints

**What it demonstrates**:
- Hybrid modeling expertise
- Production-ready code
- Alignment with Yokogawa Insilico's technology
- Ready for customer projects!

