#!/bin/bash
# Activate BBScore environment
# Usage: source activate_bbscore.sh

if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cs375_2
else
    echo "Error: conda not found"
    return 1
fi

export SCIKIT_LEARN_DATA="/Users/sebastiankane/bbscore_data"

PYTHON_PATH=$(which python)
echo "BBScore environment activated!"
echo "Python: $PYTHON_PATH"
echo "Data: $SCIKIT_LEARN_DATA"
echo ""
echo "Quick start:"
echo "  $PYTHON_PATH run.py --model resnet18 --layer layer4 --benchmark OnlineTVSDV1 --metric ridge"
