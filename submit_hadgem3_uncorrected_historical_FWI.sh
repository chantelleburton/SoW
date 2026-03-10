#!/bin/bash
#SBATCH --job-name=HadGEM3_FORCING
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=360
#SBATCH --output=logs/hot_spells_%j.out
#SBATCH --error=logs/hot_spells_%j.err

set -euo pipefail

check_status() {
    local step="$1"
    if [ $? -ne 0 ]; then
        echo "ERROR: ${step} failed"
        exit 1
    fi
}

cd /data/users/bob.potts/StateOfFires_2025-26/code
mkdir -p logs

# Initialize conda for bash (robust activation)
echo "Activating swof conda environment..."

# Temporarily disable nounset while sourcing shell init files
set +u
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# Try multiple methods to find and activate conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate swof
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate swof
else
    CONDA_BASE=$(command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null || true)
    if [ -n "${CONDA_BASE:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate sowf
    else
        echo "ERROR: Could not locate conda.sh"
        exit 1
    fi
fi
set -u

check_status "Conda environment activation"

PYTHONUNBUFFERED=1 python Historical_FWI/HadGEM3_Uncorrected_Historical_FWI.py