# Learning Weighting Confidence Networks in Semi-Supervised Algorithms using Implicit Gradients
## Environment Setup
```bash
conda create -n IML_SSL python=3.7
source activate IML_SSL
pip install -r requirements.txt
conda install -c anaconda cudatoolkit 
```
This code is tested with cudatoolkit 11 and an RTX 2070. However, with an A100 GPU on GCP, the requirements file versions are incompatible with CUDA 11. Therefore, use the Pytorch 1.9.0+Cuda11 GCP Deep Learning VM as is without any extra conda envs as it contains all packages needed out of the box. However, one might need to install tensorflow, keras, tensoboard, argparse, tqdm and ipdb if missing.