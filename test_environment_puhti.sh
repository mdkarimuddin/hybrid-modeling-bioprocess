#!/bin/bash
#SBATCH --job-name=test_hybrid_env
#SBATCH --account=project_2010726
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=test_env_%j.out
#SBATCH --error=test_env_%j.err

# On Puhti, initialize Lmod (module system)
# Lmod is used on Puhti/CSC systems
if [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
elif [ -f /appl/lmod/lmod/init/bash ]; then
    source /appl/lmod/lmod/init/bash
fi

# Set MODULEPATH if not set (needed for batch jobs)
if [ -z "$MODULEPATH" ]; then
    # Common Puhti module paths
    if [ -d /appl/modulefiles ]; then
        export MODULEPATH=/appl/modulefiles
    elif [ -d /usr/share/modulefiles ]; then
        export MODULEPATH=/usr/share/modulefiles
    fi
    # Try to get MODULEPATH from lmod
    if command -v module &> /dev/null; then
        eval $(/usr/share/lmod/lmod/libexec/lmod bash 2>/dev/null || /appl/lmod/lmod/libexec/lmod bash 2>/dev/null || true)
    fi
fi

echo "=========================================="
echo "ENVIRONMENT TEST - HYBRID MODELING PIPELINE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load required modules
echo "Loading modules..."
# Try with specific versions first (like in hyenadna setup), then fallback to default
if command -v module &> /dev/null; then
    # Try specific version first
    module load python-data/3.10-22.09 2>&1 || module load python-data 2>&1
    MODULE_PYTHON_DATA=$?
    if [ $MODULE_PYTHON_DATA -eq 0 ]; then
        echo "✅ python-data module loaded"
    else
        echo "⚠️  Failed to load python-data module (exit code: $MODULE_PYTHON_DATA)"
    fi

    # Try specific version first
    module load pytorch/2.4 2>&1 || module load pytorch 2>&1
    MODULE_PYTORCH=$?
    if [ $MODULE_PYTORCH -eq 0 ]; then
        echo "✅ pytorch module loaded"
    else
        echo "⚠️  Failed to load pytorch module (exit code: $MODULE_PYTORCH)"
    fi
else
    echo "❌ Module command not available after initialization"
    echo "Module system may not be properly configured"
fi
echo ""

# Set working directory
PIPELINE_DIR="/scratch/project_2010726/solution_data scientist/hybrid_modeling_pipeline"
cd "$PIPELINE_DIR"
echo "Working directory: $(pwd)"
echo ""

# Test Python environment
echo "=== Python Environment ==="
# Check which Python is being used
PYTHON_CMD=$(which python3)
echo "Python path: $PYTHON_CMD"
$PYTHON_CMD --version
echo "Python executable: $(readlink -f $(which python3) 2>/dev/null || which python3)"
echo ""

# Test imports
echo "=== Testing Package Imports ==="
$PYTHON_CMD << 'PYTHON_EOF'
import sys

print("Testing core packages...")
# Required packages (must be available)
required_packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'pandas': 'Pandas',
    'torch': 'PyTorch',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn'
}

# Optional packages (nice to have, but not critical)
optional_packages = {
    'sklearn': 'scikit-learn'
}

all_ok = True
for module, name in required_packages.items():
    try:
        pkg = __import__(module)
        version = getattr(pkg, '__version__', 'unknown')
        print(f"✅ {name:15s} - version {version}")
    except ImportError as e:
        print(f"❌ {name:15s} - NOT AVAILABLE: {e}")
        all_ok = False

# Test optional packages (warn but don't fail)
for module, name in optional_packages.items():
    try:
        pkg = __import__(module)
        version = getattr(pkg, '__version__', 'unknown')
        print(f"✅ {name:15s} - version {version}")
    except ImportError as e:
        print(f"⚠️  {name:15s} - NOT AVAILABLE (optional): {e}")
        print(f"   (This is OK - evaluation.py has fallback functions)")

print("")
print("=== Testing Pipeline Modules ===")
try:
    from hybrid_model import HybridModel, MechanisticModel, MLComponent
    print("✅ hybrid_model module imported successfully")
except ImportError as e:
    print(f"❌ hybrid_model import failed: {e}")
    all_ok = False

try:
    from data_processing import generate_synthetic_bioprocess_data
    print("✅ data_processing module imported successfully")
except ImportError as e:
    print(f"❌ data_processing import failed: {e}")
    all_ok = False

try:
    from training import Trainer
    print("✅ training module imported successfully")
except ImportError as e:
    print(f"❌ training import failed: {e}")
    all_ok = False

try:
    from evaluation import evaluate_model, plot_training_history
    print("✅ evaluation module imported successfully")
except ImportError as e:
    print(f"❌ evaluation import failed: {e}")
    all_ok = False

print("")
if all_ok:
    print("✅ ALL TESTS PASSED - Environment is ready!")
    sys.exit(0)
else:
    print("❌ SOME TESTS FAILED - Check errors above")
    sys.exit(1)
PYTHON_EOF

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Environment test PASSED"
    echo "You can now run the full pipeline with:"
    echo "  sbatch run_hybrid_modeling_puhti.sh"
else
    echo "❌ Environment test FAILED"
    echo "Please check the errors above and fix any missing dependencies"
fi
echo "End time: $(date)"
echo "=========================================="

