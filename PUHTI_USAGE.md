# Running Hybrid Modeling Pipeline on Puhti Supercomputer

This guide explains how to run the hybrid modeling pipeline on Puhti using SLURM batch jobs.

## Prerequisites

- Access to Puhti supercomputer
- Account: `project_2010726`
- Python environment with required packages (via `python-data` and `pytorch` modules)

## Quick Start

### 1. Test Environment

First, test that your environment is set up correctly:

```bash
cd "/scratch/project_2010726/solution_data scientist/hybrid_modeling_pipeline"
sbatch test_environment_puhti.sh
```

This will:
- Load the `python-data` and `pytorch` modules
- Check that all required Python packages are available
- Verify that pipeline modules can be imported
- Complete in ~1-2 minutes

Check the output:
```bash
# After job completes, check the output file
cat test_env_<JOB_ID>.out
```

### 2. Run Full Pipeline

Once the environment test passes, run the complete pipeline:

```bash
sbatch run_hybrid_modeling_puhti.sh
```

This will:
- Generate synthetic bioprocess data
- Train the hybrid model
- Evaluate performance
- Generate visualization plots
- Save model and results

**Expected runtime**: ~30-60 minutes (depending on system load)

## Monitoring Jobs

### Check Job Status

```bash
squeue -u $USER
```

### View Job Output (while running)

```bash
# Replace <JOB_ID> with your actual job ID
tail -f hybrid_modeling_<JOB_ID>.out
```

### View Job Errors

```bash
tail -f hybrid_modeling_<JOB_ID>.err
```

## Job Configuration

### Resource Allocation

The batch script (`run_hybrid_modeling_puhti.sh`) is configured with:

- **Partition**: `small`
- **Time**: 2 hours (02:00:00)
- **CPUs**: 4 cores
- **Memory**: 16 GB
- **Account**: `project_2010726`

### Adjusting Resources

If you need more resources, edit `run_hybrid_modeling_puhti.sh`:

```bash
# For longer training or larger datasets:
#SBATCH --time=04:00:00      # 4 hours
#SBATCH --cpus-per-task=8     # 8 CPUs
#SBATCH --mem=32G             # 32 GB memory

# For GPU training (if available):
#SBATCH --gres=gpu:v100:1     # Request 1 V100 GPU
```

## Output Files

After successful completion, you'll find:

### In `outputs/` directory:
- `training_history.png` - Training/validation loss curves
- `predictions.png` - Time series predictions
- `prediction_scatter.png` - Predicted vs. true scatter plots
- `metrics_comparison.png` - Hybrid vs. mechanistic model comparison
- `final_model.pt` - Trained model checkpoint
- `checkpoints/best_model.pt` - Best model during training

### Log Files:
- `hybrid_modeling_<JOB_ID>.out` - Standard output
- `hybrid_modeling_<JOB_ID>.err` - Standard error

## Troubleshooting

### Issue: Module not found

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Make sure both modules are loaded:
```bash
module load python-data
module load pytorch
```

If PyTorch is still not available:
1. Check available PyTorch versions: `module avail pytorch`
2. Try a specific version: `module load pytorch/2.0`
3. Check Puhti documentation for PyTorch module availability

### Issue: Matplotlib backend error

**Error**: `UserWarning: Matplotlib is currently using agg, which is a non-GUI backend`

**Solution**: This is expected and correct for batch jobs. The script automatically sets `MPLBACKEND=Agg` for non-interactive use.

### Issue: Out of memory

**Error**: `RuntimeError: CUDA out of memory` or `MemoryError`

**Solution**: 
1. Reduce batch size in `example_usage.py`:
   ```python
   batch_size = 16  # Instead of 32
   ```
2. Reduce number of experiments:
   ```python
   data, time, params = generate_synthetic_bioprocess_data(
       n_experiments=10,  # Instead of 20
       ...
   )
   ```
3. Request more memory in batch script:
   ```bash
   #SBATCH --mem=32G
   ```

### Issue: Job times out

**Error**: Job killed before completion

**Solution**: Increase time limit:
```bash
#SBATCH --time=04:00:00  # 4 hours instead of 2
```

### Issue: Import errors

**Error**: `ImportError: cannot import name 'HybridModel'`

**Solution**: 
1. Make sure you're in the correct directory
2. Check that all Python files are present
3. Run the environment test first

## Customizing for Your Data

### Using Your Own Data

1. Modify `example_usage.py` to load your data:
   ```python
   from data_processing import load_real_data
   
   # Load your data
   data = load_real_data('path/to/your/data.csv')
   ```

2. Adjust data preprocessing as needed

3. Update model parameters to match your process

### Adjusting Model Parameters

Edit `example_usage.py`:

```python
# Mechanistic parameters for your specific process
mechanistic_params = {
    'mu_max': 0.6,   # Adjust based on your cell line
    'Ks': 0.15,      # Adjust based on your substrate
    'Yxs': 0.55,
    'Yps': 0.35,
    'qp_max': 0.12
}

# Model architecture
model = HybridModel(
    mechanistic_params=mechanistic_params,
    ml_input_dim=5,      # Add more features (pH, T, etc.)
    ml_hidden_dim=128,  # Larger network
    ml_num_layers=3,    # Deeper network
    use_residual_learning=True
)
```

## Example Workflow

```bash
# 1. Navigate to pipeline directory
cd "/scratch/project_2010726/solution_data scientist/hybrid_modeling_pipeline"

# 2. Test environment
sbatch test_environment_puhti.sh

# 3. Wait for test to complete, then check results
cat test_env_<JOB_ID>.out

# 4. If test passes, run full pipeline
sbatch run_hybrid_modeling_puhti.sh

# 5. Monitor job
squeue -u $USER

# 6. After completion, check results
ls -lh outputs/
cat hybrid_modeling_<JOB_ID>.out
```

## Advanced Usage

### Running Multiple Experiments

Create a script to run multiple configurations:

```bash
#!/bin/bash
# run_multiple_experiments.sh

for hidden_dim in 32 64 128; do
    for lr in 0.001 0.0005 0.0001; do
        # Modify example_usage.py with these parameters
        # Then submit job
        sbatch run_hybrid_modeling_puhti.sh
    done
done
```

### Using GPU (if available)

Modify `run_hybrid_modeling_puhti.sh`:

```bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
```

And in `example_usage.py`, the device will automatically be set to 'cuda' if available.

## Getting Help

- Check Puhti documentation: https://docs.csc.fi/computing/overview/
- Check SLURM documentation: https://slurm.schedmd.com/
- Review job logs for specific error messages

## Notes

- Both `python-data` and `pytorch` modules need to be loaded
- The `python-data` module includes most scientific Python packages (NumPy, SciPy, Pandas, etc.)
- The `pytorch` module provides PyTorch for deep learning
- For GPU training, you may need to load additional modules or use GPU partition
- Always test with small datasets first before running large experiments

