# A Variational Autoencoder Implementation

## Get started

This implementation is based on PyTorch. All code is in `main.py`, including both training and evaluation parts. To run the code, you need to first install required dependencies:

```shell
python3 -m pip install -r requirements.txt
```

Then you can run the code:

```shell
python3 main.py
```

If you have at least one NVIDIA GPU on your machine and have installed CUDA, the program will be executed on GPU. We highly recommend you to use GPU since it will reduce the training time tremendously.

## Reproducibility

We have tried many different random seeds and chosen a good one. The random seed have been set by `torch.manual_seed()` in `main.py` , the results should be the same as the demonstrated images.
