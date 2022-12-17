# A Variational Autoencoder Implementation

## Get started

This implementation is based on PyTorch. All the code is in `main.py`, including both training and evaluation parts. To run the code, you need to first install required dependencies:

```shell
python3 -m pip install -r requirements.txt
```

And you also need to download the training images from [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), we didn't prepare the dataset in repository since it's too large (about 757MB). Put all the image files into `input/all-dogs` directory then the code will be able to access to it, the directory structure will be like:

```
.
├── LICENSE
├── README.md
├── input
│   └── all-dogs
│       ├── n02085620_199.jpg
│       ├── n02085620_242.jpg
│       ├── n02085620_7.jpg
│       └── ...
├── main.py
└── requirements.txt
```

One thing should be noticed is that the decompressed folder of Stanford Dogs Dataset includes multiple subfolders corresponding to different categories, which is used for classification tasks. However, in our case, we only need to reconstruct the image and there is no need to perform classification, so this is actually an unsupervised learning which means we don't need annotations and we can put all the images with different categories into one folder to train. So we recommend you to use another dataset provided by [Kaggle](https://www.kaggle.com/competitions/generative-dog-images/data?select=all-dogs.zip) for convenience, its data is the same as Stanford's except that it has put all the images into one folder.

Then you can run the code, all the output images will be saved in the `output` directory for the convenience for Linux Server developers as server side normally doesn't have GUI to display images.

```shell
python3 main.py
```

If you have at least one NVIDIA GPU on your machine and have installed CUDA, the program will be executed on GPU. It's **highly recommended** to use GPU since it will reduce the training time tremendously.

## Results demonstration

Reconstruct one image after each iteration of epoch:

![Epochs](./figures/Epochs.png)

The overall reconstruction results on multiple images:

![Checks](./figures/Checks.png)

Walk in the latent space from one dog to another:

![Walks](./figures/Walks.png)

## Reproducibility

We have tried many different random seeds and chosen a good one. The random seed has been set by `torch.manual_seed()` in `main.py` , the running results should be the same as the demonstrated images.
