# gan_practice
dcgan,wgan and improved wgan implementation by tensorflow

# how to use
1.download MNIST data from http://yann.lecun.com/exdb/mnist/ and unzip data to some dir
  for example: data
                   \_10k-images-idx3-ubyte
                   \_train-images-idx3-ubyte

# training
python dcgan.py or wgan.py or wgan_gp.py \
                --data_path=data \
                --mode=train \
                --model_dir=models \
                --batch_size=100 \
                --epoch_num=100 \
                --rand_dim=128

# generating
python dcgan.py or wgan.py or wgan_gp.py \
                --data_path=data \
                --mode=infer \
                --model_dir=models \
                --batch_size=100 \
                --epoch_num=100 \
                --rand_dim=128

# some results
1.dcgan 100 epoch
2.wgan 100 epoch
3.improved wgan epoch

# reference
* Generative Adversarial Nets
* Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
* Wasserstein GAN
* Improved Training of Wasserstein GANs
* https://github.com/carpedm20/DCGAN-tensorflow
* https://github.com/shekkizh/WassersteinGAN.tensorflow
* https://github.com/igul222/improved_wgan_training


