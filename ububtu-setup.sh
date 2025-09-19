#!/bin/bash
set -e

# ====== CONFIG ======
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="pinto"
PYTHON_VERSION="3.10"
# ====================



echo ">>> Downloading Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

echo ">>> Installing Miniconda to $CONDA_DIR..."
bash ~/miniconda.sh -b -p $CONDA_DIR
rm ~/miniconda.sh

echo ">>> Initializing Conda..."
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
$CONDA_DIR/bin/conda init bash

conda tos accept
echo ">>> Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo ">>> Activating environment and installing base packages..."
conda activate $ENV_NAME

pip3 install 'tensorflow[and-cuda]' numpy h5py pyDOE matplotlib pandas wandb

cd Advection
python3 Code/PINTO/PINTO_model.py