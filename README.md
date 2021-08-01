# Implicit Gradients for Learning Weighting Confidence Networks in Semi-Supervised Algorithms
## Environment Setup
```bash
conda create -n IML_SSL python=3.7
source activate IML_SSL
pip install -r requirements.txt
conda install -c anaconda cudatoolkit 
```
This code is tested with cudatoolkit 11 and an RTX 2070. However, with an A100 GPU on GCP, the requirements file versions are incompatible with CUDA 11. Therefore, use the following command to update the version:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```