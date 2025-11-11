#!/bin/bash
# Azure ML Compute Setup Script for edu_ai_library

# Update system
sudo apt update && sudo apt upgrade -y

# Install Miniforge (Anaconda base)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge3.sh
bash Miniforge3.sh -b -p $HOME/miniforge3
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

# Create and activate environment
conda create -y -n edu_ai_library python=3.10
conda activate edu_ai_library

# Install dependencies
conda install -y pandas numpy scikit-learn fastapi uvicorn streamlit python-pptx faker joblib requests

# Optional: install Azure SDKs
pip install azure-ai-ml azure-identity openai langchain

# Clone your GitHub repo
cd ~
git clone https://github.com/kaqugh/Edu_AI_Library.git

echo "✅ Environment setup complete"
