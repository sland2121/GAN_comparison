Citations:

This code was structured after the code here:
-https://github.com/tolstikhin/adagan

We referred to the following implementations for implementing the GAN algorithms and metrics:

-AdaGAN: https://github.com/tolstikhin/adagan
-WGAN: https://github.com/cameronfabbri/Wasserstein-GAN-Tensorflow
-VEEGAN: https://github.com/akashgit/VEEGAN
-Unrolled GAN: https://github.com/tolstikhin/adagan, https://github.com/poolio/unrolled_gan



Goal:
This code supports running experiments on synthetic (2d grid, 2d ring, high dimensional) and real datasets (MNIST) for various GAN algorithms
as described in our 6.883 Final Project Report.

Supported GAN algorithms include: AdaGAN, VEEGAN, Wasserstein GAN, Unrolled GAN.

Command to train and sample a GAN:

	python run.py <INSERT DATASET_NAME i.e., 2d_ring, 2d_grid, hd, mnist> <INSERT GAN NAME i.e., AdaGAN, WGAN, UnrolledGAN, VEEGAN>

See Metrics.py for example on computing metrics for a GAN and dataset.




