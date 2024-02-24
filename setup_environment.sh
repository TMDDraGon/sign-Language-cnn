#!/bin/bash

# Download and install Miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc

# Create a new conda environment named tf_env with Python 3.8
conda create -y -n tf_env python=3.8
conda activate tf_env 

# Install TensorFlow-GPU version 2.2
conda install -y tensorflow-gpu=2.2

# Install specific versions of Python packages
pip install scikit-learn==0.24.0
pip install seaborn==0.11.1
pip install matplotlib==3.3.3
pip install tensorflow==2.3.*
pip install pandas==1.2.4
pip install scipy==1.7.3

# pip install pexpect==4.9
# pip install typing-extensions==4.9