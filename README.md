# GAN_comparison

# Contributors:

Anastasiya Belyaeva, Sayeri Lala, Molei Liu, Maha Shady


# Goal:
This code supports running experiments on synthetic (2d grid, 2d ring, high dimensional) and real datasets (MNIST) for various GAN algorithms as described in our 6.883 Final Project Report.

Supported GAN algorithms include: AdaGAN, VEEGAN, Wasserstein GAN, Unrolled GAN.

Command to train and sample a GAN:

	python run.py <INSERT DATASET_NAME i.e., 2d_ring, 2d_grid, hd, mnist> <INSERT GAN NAME i.e., AdaGAN, WGAN, UnrolledGAN, VEEGAN>


# Datasets:
Zip files of the synthetic and real datasets are available at the following links:

-synthetic: https://www.dropbox.com/s/6enn2v1kfwcw9qv/synthetic_datasets.zip?dl=0

-MNIST: https://www.dropbox.com/s/0hhrn6qldgzv3ef/MNIST-data.zip?dl=0


# Citations:

This code was structured after the code here:
-https://github.com/tolstikhin/adagan

We referred to and used code from the following implementations for implementing the GAN algorithms and metrics:

-AdaGAN: https://github.com/tolstikhin/adagan

-WGAN: https://github.com/cameronfabbri/Wasserstein-GAN-Tensorflow

-VEEGAN: https://github.com/akashgit/VEEGAN

-Unrolled GAN: https://github.com/tolstikhin/adagan, https://github.com/poolio/unrolled_gan




