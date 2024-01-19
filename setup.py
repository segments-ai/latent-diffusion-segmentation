from setuptools import setup, find_packages
from ldmseg.version import __version__

setup(
  name='ldmseg',
  packages=find_packages(),
  version=__version__,
  license='Attribution-NonCommercial 4.0 International',
  description='A simple latent diffusion framework for panoptic segmentation and mask inpainting',
  author='Wouter Van Gansbeke',
  url='https://github.com/segments-ai/latent-diffusion-segmentation',
  long_description_content_type='text/markdown',
  keywords=[
    'artificial intelligence',
    'computer vision',
    'generative models',
    'segmentation',
  ],
  install_requires=[
    'torch',
    'torchvision',
    'einops',
    'diffusers',
    'transformers',
    'xformers',
    'accelerate',
    'timm',
    'scipy',
    'opencv-python',
    'pyyaml',
    'easydict',
    'hydra-core',
    'termcolor',
    'wandb',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.11',
  ],
)
