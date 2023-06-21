# Clustering satellite images of Accra

based on the publication titled "Phenotyping urban built and natural environments with high-resolution
satellite images and unsupervised deep learning" published in STOTEN, 2023.


The clustering algorithm was adapted from DeepCluster by Caron (2018), which was published by Facebook research and
is also openly available, was run on 3 RTX6000 GPUs, 72GB memory and a runtime of approximately 24 h.

https://github.com/facebookresearch/deepcluster

## Requirements
- Python 3
- the SciPy and scikit-learn packages
- a PyTorch install version 0.1.8 (pytorch.org)
- CUDA 8.0
- a Faiss install (Faiss)
- The ImageNet dataset (which can be automatically downloaded by recent version of torchvision)
