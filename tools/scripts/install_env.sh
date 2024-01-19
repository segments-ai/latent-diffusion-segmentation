#!/bin/bash
miniconda3_path=${1-'/home/ubuntu/datasets/miniconda3'}

# check if miniconda3_path exists
if [ ! -d "$miniconda3_path" ]; then
    echo "Error: miniconda3_path does not exist."
    exit 1
fi

# check we are running this from ldmseg root directory
if [ ! -d "tools" ]; then
    echo "Error: tools directory does not exist. You're not in the repo's root directory."
    exit 1
fi

# prints
echo "Installing environment, using miniconda path ..."
echo ${miniconda3_path}/etc/profile.d/conda.sh

# install environment
. ${miniconda3_path}/etc/profile.d/conda.sh  # initialize conda
conda create -n ldmseg_env python -y
conda activate ldmseg_env

# install packages
python -m pip install -e .                                           
pip install git+https://github.com/facebookresearch/detectron2.git   # detectron2 
pip install git+https://github.com/cocodataset/panopticapi.git       # pq evaluation   (optional)
pip install flake8 black                                             # code formatting (optional)
