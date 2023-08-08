# SimCLR Implementation with Pytorch and Applied to the Kaggle Flowers Dataset

[SimCLR](https://arxiv.org/abs/2002.05709) model which is able to produce image representations using contrastive learning. The model is implemented with PyTorch and uses the Flowers dataset from Kaggle.

Everything is implemented in the notebook [`simclr_pytorch_flowers.ipynb`](./simclr_pytorch_flowers.ipynb).

## Setup

I used my NVIDIA RTX 3060 as an eGPU on Windows 10.

To check the GPU usage - RAM and GPU processors:
	
	Shell:
		(once)
		nvidia-smi.exe
		(everz 10 seconds)
		nvidia-smi.exe -l 10

	Notebook:
		!nvidia-smi

Basic environment installation with Pytorch:

```bash
# Crate env: requirements in conda.yaml
# This packages are the basic for Pytorch-CUDA usage
# Additionally Tensorflow/Keras is included for CPU
conda env create -f conda.yaml
conda activate siam

# Pytorch: Windows + CUDA 11.7
# Update your NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
# I have version 12.1, but it works with older versions, e.g. 11.7
# Check your CUDA version with: nvidia-smi.exe
# In case of any runtime errors, check vrsion compatibility tables:
# https://github.com/pytorch/vision#installation
# The default conda installation command DID NOT WORK
# But the following pip install command DID WORK
python -m pip install torch==1.13+cu117 torchvision==0.14+cu117 torchaudio torchtext==0.14 --index-url https://download.pytorch.org/whl/cu117

# Pytorch: Mac / Windows CPU (not necessary if the previous line is executed)
python -m pip install torch torchvision torchaudio
```

Finally, dump a `requirements.txt` with all dependencies (with used versions):

```bash
# Dump installed libraries in pip format
python -m pip list --format=freeze > requirements.txt
```

## Links

Essential sources of the implementation

- [SimCLR](https://arxiv.org/abs/2002.05709)
- [Official SimCLR repository](https://github.com/google-research/simclr)
- [Amit Chaudhary's blog on SimCLR](https://amitness.com/2020/03/illustrated-simclr/)
- [Sayak Paul's blog on SimCLR](https://wandb.ai/sayakpaul/simclr/reports/Towards-Self-Supervised-Image-Understanding-with-SimCLR--VmlldzoxMDI5NDM)
- [sthalles' SimCLR implementation in PyTorch](https://github.com/sthalles/SimCLR)
- [SimCLR at Kaggle](https://www.kaggle.com/code/aritrag/simclr)
- [**Flowers Dataset at Kaggle**](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)

For more general and related contents, consider visiting:

- [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity)
- [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity)
- [pyimagesearch_tutorials](https://github.com/mxagar/pyimagesearch_tutorials)

## Authorship

Mikel Sagardia, 2023.  
No guarantees.

Also, check the authors of the links.
