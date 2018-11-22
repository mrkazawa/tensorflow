# Machine Learning HW 2 - No 9

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()
[![dependency](https://img.shields.io/badge/dependency-tensorflow-orange.svg)]()

This repository contains our code to answer the Machine Learning class Homework 2 No 9. We will perform a **DCGAN (Deep Convolutional Generative Adversarial Network)** to generate fake image data by training them against [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Our code is a modification based on the code available from the [Tensorflow Tutorial](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb) site.

## Prerequisites

We use TensorFlow version 1.11 to run the experiment. So, you need to install TensorFlow before running this python file. We also use imageio to generate gif image. Thus, you need to install it as well.

```
pip install tensorflow=1.11.0
pip install imageio
```

## How to run / train

You can start running the program with the training. It will take some time depending on your machine. *(took me around 1 hour with 1080Ti card)*

```shell
cd YOUR_DIR
python normal.py # to run scenario 1
python weak_g.py # to run scenario 2
python weak_d.py # to run scenario 3
```

## Data Normalization

The first thing to do is obviously to insert the dataset into our code. However, before we start put the dataset into the training process, we need to perform several data normalization tasks, described as follows:

### 1 - Reshaping the dimension of the image data

Transforming the image dataset to have 3D dimension to be trained by the convolution layer. This particular dimension is important because it needs to be compatible with the Model that we are trying to built. Otherwise, the program will error. Original image is in 2D 28 by 28 format, we add 1 into the 'depth' because it is greyscale image. Thus the new dimension will be 3D (28, 28, 1) As an additional information, colored image will have the 'depth' of 3 becuase it contains RGB value.

```python
def reshape_images(train_images):
    return train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
```

### 2 - Centering the image

We are centering the image data to the range of -1 to 1. This process is necessary to the training process run more efficient and faster. If we are using the original data (between 0 to 255), the model will converge very slowly.

```python
def centering_images(train_images):
    return (train_images - 127.5) / 127.5
```

### 3 - Make a new custom dataset object

When we are doing GAN with the DCGAN method, we are only using the image data to train our gan. Therefore, we create a new Tensorflow dataset object that will only include the x_train data and excluding the y_train, x_test y_test data.

```python
def create_custom_dataset(data):
    # Create custom dataset that only contains image data
    return tf.data.Dataset.from_tensor_slices(data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

## The DCGAN Implementation

In this section we are going to run and analyze our DCGAN code towards the FASHION MNIST dataset. To be able to investigate the behaviour of the generator and the discriminator model during the training, we do some modification to the code to plot the history of the loss during the training. Also, we divide our test into several scenarios:

* Using initial generator and discriminator model
* Using weaker generator model
* Using weaker discriminator model

### 1 - Using initial generator and discriminator model

What we mean by initial generator and discriminator model is the models that are implemented in the earlier code. We do not modify the structure of the models in this scenario. Based on that code, the **generator model** layer and configuration looks something like this:

* Dense (7 x 7 x 64 units) -> BatchNorm -> ReLU
* Conv2DTranspose (64 features) -> BatchNorm -> ReLU
* Conv2DTranspose (32 features) -> BatchNorm -> ReLU
* Conv2DTranspose (1 feature) --> tanh

Output from Tensorflow `model.summary()`
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                multiple                  313600
_________________________________________________________________
batch_normalization (BatchNo multiple                  12544
_________________________________________________________________
conv2d_transpose (Conv2DTran multiple                  102400
_________________________________________________________________
batch_normalization_1 (Batch multiple                  256
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr multiple                  51200
_________________________________________________________________
batch_normalization_2 (Batch multiple                  128
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr multiple                  800
=================================================================
Total params: 480,928
Trainable params: 474,464
Non-trainable params: 6,464
_________________________________________________________________
```

Meanwhile, the **discriminator model** layer and configuration are something like this:

* Conv2D (64 features) -> LeakyReLU
* Dropout
* Conv2D (128 features) -> LeakyReLU
* Dropout
* Flatten
* Dense

Output from Tensorflow `model.summary()`

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              multiple                  1664
_________________________________________________________________
conv2d_1 (Conv2D)            multiple                  204928
_________________________________________________________________
dropout (Dropout)            multiple                  0
_________________________________________________________________
flatten (Flatten)            multiple                  0
_________________________________________________________________
dense_1 (Dense)              multiple                  6273
=================================================================
Total params: 212,865
Trainable params: 212,865
Non-trainable params: 0
_________________________________________________________________
```

We run this scenario in 300 epochs **(it took around 1 hour with 1080Ti card)**. Below is the chart of the generator and discriminator loss during the training.
![Normal Loss](img/normal_loss.png?raw=true "normal_loss")
We can see from the figure that at the early stage (`epoch 1-50`), the generator loss is high because the generator still learning to generate images and the discriminator loss is low because it can easily differentiate between the fake and real images. However both lossess are starting to twisted around `epoch 75`. At this time, the discriminator loss is higher than the generator loss. As the generator started to be good at producing image, the loss will be getting better while the discriminator loss will be worse bacause it is now more difficult to differentiate between the fake and real images. Based on our experiment, both model start to converge around `epoch 125`.

### 2 - Using weaker discriminator model

In this scenario, we try to modify the discriminator model so that it is really weak and can only produce small number of trainable parameters. This should make the discriminator **really bad** at differentiating between fake and real images. We want to see if this model can still generate good enough fake images. The **weak discriminator model** layer and configuration for this scenario looks something like this:

* Conv2D (16 features) -> LeakyReLU
* Dropout
* Conv2D (32 features) -> LeakyReLU
* Dropout
* Flatten
* Dense

Output from Tensorflow `model.summary()`
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              multiple                  416
_________________________________________________________________
conv2d_1 (Conv2D)            multiple                  12832
_________________________________________________________________
average_pooling2d (AveragePo multiple                  0
_________________________________________________________________
dropout (Dropout)            multiple                  0
_________________________________________________________________
flatten (Flatten)            multiple                  0
_________________________________________________________________
dense_1 (Dense)              multiple                  1569
=================================================================
Total params: 14,817
Trainable params: 14,817
Non-trainable params: 0
_________________________________________________________________
```

We use the same **generatore model** as in Scenario 1. Then, we run this scenario in 300 epochs **(it took around 1 hour with 1080Ti card)**. Below is the chart of the generator and discriminator loss during the training.
![Weak D Loss](img/weak_disc_loss.png?raw=true "weak_d_loss")
We can see from the figure that at the early stage (`epoch 1-50`), the discriminator loss is already high due to weak classification ability. This weak discriminator affects the ability of the generator to produce good image. We can see from the figure that at this early stage, the loss distribution of the generator model is not as diverse as the early stage at the Scenario 1. This hinders the growth of the generator model. Lastly, based on our experiment, both model start to converge early (compared to Scenario 1) around `epoch 75`.

### 3 - Using weaker generator model

In this scenario, we try to perform the opposite test from the Scenario 2, we want modify the generator model so that it is really weak and can only produce small number of trainable parameters. This should make the generator **really bad** at generating fake images. We want to see whether using this combination, the model can still generate good enough fake images. The **weak generator model** layer and configuration for this scenario looks something like this:

* Dense (7 x 7 x 16 units) -> BatchNorm -> ReLU
* Conv2DTranspose (8 features) -> BatchNorm -> ReLU
* Conv2DTranspose (1 feature) --> tanh

Output from Tensorflow `model.summary()`
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              multiple                  78400
_________________________________________________________________
batch_normalization (BatchNo multiple                  3136
_________________________________________________________________
conv2d_transpose (Conv2DTran multiple                  3200
_________________________________________________________________
batch_normalization_1 (Batch multiple                  32
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr multiple                  200
_________________________________________________________________
dropout_1 (Dropout)          multiple                  0
=================================================================
Total params: 84,968
Trainable params: 83,384
Non-trainable params: 1,584
_________________________________________________________________
```
We use the same **discriminator model** as in Scenario 1. Then, we run this scenario in 300 epochs **(it took around 1 hour with 1080Ti card)**. Below is the chart of the generator and discriminator loss during the training.
![Weak G Loss](img/weak_gen_loss.png?raw=true "weak_g_loss")
We can see from the figure that at the early stage (`epoch 1-50`), the generator loss is high because it can only generate bad fake images. Moreover, based on our experiment, this scenario is harder to reach a linear convergence. Even at `epoch 300`, the generator loss still tend to go to a slightly higher value. However, the discriminator loss graph shows that it converge around `epoch 150`.

## Results Comparison

Normal (Epoch 1 - 300) | Weak Discriminator (Epoch 1 - 300) | Weak Generator (Epoch 1 - 300)
:---: | :---: | :---:
![Normal GIF](img/normal.gif?raw=true "normal_gif") | ![Weak D GIF](img/weak_disc.gif?raw=true "weak_d_gif") | ![Weak G GIF](img/weak_gen.gif?raw=true "weak_g_gif") 

Normal (Epoch 25) | Weak Discriminator (Epoch 25) | Weak Generator (Epoch 25)
:---: | :---: | :---:
![Normal 25](img/normal_25.png?raw=true "normal_25") | ![Weak D 25](img/weak_disc_25.png?raw=true "weak_d_25") | ![Weak G 25](img/weak_gen_25.png?raw=true "weak_g_25")

Normal (Epoch 50) | Weak Discriminator (Epoch 50) | Weak Generator (Epoch 50)
:---: | :---: | :---:
![Normal 50](img/normal_50.png?raw=true "normal_50") | ![Weak D 50](img/weak_disc_50.png?raw=true "weak_d_50") | ![Weak G 50](img/weak_gen_50.png?raw=true "weak_g_50")

Normal (Epoch 75) | Weak Discriminator (Epoch 75) | Weak Generator (Epoch 75)
:---: | :---: | :---:
![Normal 75](img/normal_75.png?raw=true "normal_75") | ![Weak D 75](img/weak_disc_75.png?raw=true "weak_d_75") | ![Weak G 75](img/weak_gen_75.png?raw=true "weak_g_75")

Normal (Epoch 100) | Weak Discriminator (Epoch 100) | Weak Generator (Epoch 100)
:---: | :---: | :---:
![Normal 100](img/normal_100.png?raw=true "normal_100") | ![Weak D 100](img/weak_disc_100.png?raw=true "weak_d_100") | ![Weak G 100](img/weak_gen_100.png?raw=true "weak_g_100")

Normal (Epoch 150) | Weak Discriminator (Epoch 150) | Weak Generator (Epoch 150)
:---: | :---: | :---:
![Normal 150](img/normal_150.png?raw=true "normal_150") | ![Weak D 150](img/weak_disc_150.png?raw=true "weak_d_150") | ![Weak G 150](img/weak_gen_150.png?raw=true "weak_g_150")

Normal (Epoch 300) | Weak Discriminator (Epoch 300) | Weak Generator (Epoch 300)
:---: | :---: | :---:
![Normal 300](img/normal_300.png?raw=true "normal_300") | ![Weak D 300](img/weak_disc_300.png?raw=true "weak_d_300") | ![Weak G 300](img/weak_gen_300.png?raw=true "weak_g_300")

## Some Takeaways

After running this experiment, we can take several lessons and takeaways:

* The loss of generator model will affect the loss of discriminator model and vice versa. They are two networks that are trying to optimize a different and opposing loss functions. So, when the loss of generator ascends, the loss of discriminator will descends. Their losses push against each other.
* The generator and discriminator, however bad they are, will converge at a given time. We have trained our generator and discriminator in several scenarios with one being weaker against the other. They are going to converge eventually.
* After the training, both of the weaker scenarios (#2 and #3) can still generate observable fake images. The weaker discriminator will have the tendency to generate bright fake images with many white pixels. Meanwhile, the weaker generator will have the opposite tendency by generating darker fake images with many black pixels.
* GAN is similar to the Neural Network in general. In order to achive the best result, we have to configure the model. Applying the suitable hyperparameters such as choosing the correct number of layers and the right type layers is important. The parameters for each layers are also important factor.

## Authors

**Oktian Yustus Eko** - *Initial Work* - [mrkazawa](https://github.com/mrkazawa)