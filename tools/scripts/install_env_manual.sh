#  Example script to install environment:
#  adapt paths accordingly

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

conda create -n pytorch_cuda12 python=3.11
conda activate pytorch_cuda12
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip install torch torchvision
pip install xformers einops diffusers transformers accelerate timm scipy opencv-python
pip install pyyaml easydict hydra-core termcolor wandb

python -m pip install -e .

conda install -c conda-forge gcc=10.4
conda install -c conda-forge gxx=10.4
export CC=/scratch/wouter_vangansbeke/miniconda3/envs/pytorch_cuda12/bin/gcc
export CXX=/scratch/wouter_vangansbeke/miniconda3/envs/pytorch_cuda12/bin/g++

pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/cocodataset/panopticapi.git
