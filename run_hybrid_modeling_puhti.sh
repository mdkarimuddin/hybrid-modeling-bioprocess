#!/bin/bash
#SBATCH --job-name=hybrid_modeling
#SBATCH --account=project_2010726
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=hybrid_modeling_%j.out
#SBATCH --error=hybrid_modeling_%j.err

# Print job info
echo "=========================================="
echo "HYBRID MODELING PIPELINE - PUHTI BATCH JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="
echo ""

# On Puhti, initialize Lmod (module system)
# Lmod is used on Puhti/CSC systems
if [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
elif [ -f /appl/lmod/lmod/init/bash ]; then
    source /appl/lmod/lmod/init/bash
fi

# Load required modules
echo "Loading modules..."
if command -v module &> /dev/null; then
    # Try specific versions first (like in hyenadna setup), then fallback to default
    module load python-data/3.10-22.09 2>&1 || module load python-data 2>&1
    module load pytorch/2.4 2>&1 || module load pytorch 2>&1
    echo "✅ python-data module loaded"
    echo "✅ pytorch module loaded"
else
    echo "❌ Module command not available"
    exit 1
fi
echo ""

# Set working directory
PIPELINE_DIR="/scratch/project_2010726/solution_data scientist/hybrid_modeling_pipeline"
cd "$PIPELINE_DIR"
echo "Working directory: $(pwd)"
echo ""

# Find Python from modules
echo "=== Environment Check ==="
# After loading pytorch module, Python should be at /appl/soft/ai/wrap/pytorch-2.4/bin/python3
# Check if that path exists, otherwise try to find it
if [ -f /appl/soft/ai/wrap/pytorch-2.4/bin/python3 ]; then
    PYTHON_CMD=/appl/soft/ai/wrap/pytorch-2.4/bin/python3
elif [ -f /appl/soft/ai/python-data/3.10-22.09/bin/python3 ]; then
    PYTHON_CMD=/appl/soft/ai/python-data/3.10-22.09/bin/python3
else
    # Try to find Python in PATH (modules should update PATH)
    PYTHON_CMD=$(which python3)
    # If still system Python, search for module Python
    if [ "$PYTHON_CMD" = "/usr/bin/python3" ] || [ "$PYTHON_CMD" = "/usr/libexec/platform-python3.6" ]; then
        # Search in common module locations
        for base in /appl/soft/ai/wrap/pytorch-2.4 /appl/soft/ai/python-data/3.10-22.09; do
            if [ -f "$base/bin/python3" ]; then
                PYTHON_CMD="$base/bin/python3"
                break
            fi
        done
    fi
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check if required Python packages are available
echo "Checking Python packages..."
$PYTHON_CMD << EOF
import sys
# Required packages
required_packages = ['numpy', 'scipy', 'pandas', 'torch', 'matplotlib', 'seaborn']
# Optional packages
optional_packages = ['sklearn']

missing_required = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} available")
    except ImportError:
        print(f"❌ {pkg} NOT available")
        missing_required.append(pkg)

# Check optional packages
for pkg in optional_packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} available")
    except ImportError:
        print(f"⚠️  {pkg} NOT available (optional - evaluation.py has fallback functions)")

if missing_required:
    print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
    print("You may need to install them or use a different Python environment")
    sys.exit(1)
else:
    print("\n✅ All required packages are available")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Environment check failed"
    exit 1
fi

echo ""
echo "=== Starting Hybrid Modeling Pipeline ==="
echo ""

# Configure matplotlib for non-interactive backend (required for batch jobs)
export MPLBACKEND=Agg

# Ensure we use the module Python (re-check if needed)
if [ ! -f "$PYTHON_CMD" ] || [ "$PYTHON_CMD" = "/usr/bin/python3" ]; then
    if [ -f /appl/soft/ai/wrap/pytorch-2.4/bin/python3 ]; then
        PYTHON_CMD=/appl/soft/ai/wrap/pytorch-2.4/bin/python3
    fi
fi

echo "Running pipeline with: $PYTHON_CMD"
echo ""

# Run the example usage script with unbuffered output
$PYTHON_CMD -u example_usage.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "JOB SUMMARY"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Job completed successfully"
    echo ""
    echo "📁 OUTPUT FILES:"
    if [ -d "outputs" ]; then
        ls -lh outputs/ 2>/dev/null | while read line; do 
            echo "  $line"
        done
    else
        echo "  No outputs directory found"
    fi
    
    echo ""
    echo "📊 Check the following for results:"
    echo "  - outputs/training_history.png: Training curves"
    echo "  - outputs/predictions.png: Model predictions"
    echo "  - outputs/prediction_scatter.png: Prediction accuracy"
    echo "  - outputs/metrics_comparison.png: Model comparison"
    echo "  - outputs/final_model.pt: Trained model checkpoint"
else
    echo "❌ Job failed with exit code $EXIT_CODE"
    echo ""
    echo "📋 Check error log for details:"
    echo "  /scratch/project_2010726/solution_data scientist/hybrid_modeling_pipeline/hybrid_modeling_${SLURM_JOB_ID}.err"
fi

echo ""
echo "📁 Full logs available at:"
echo "  stdout: hybrid_modeling_${SLURM_JOB_ID}.out"
echo "  stderr: hybrid_modeling_${SLURM_JOB_ID}.err"
echo "=========================================="

