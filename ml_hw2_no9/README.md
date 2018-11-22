# Machine Learning HW 2 - No 9

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()
[![dependency](https://img.shields.io/badge/dependency-tensorflow-orange.svg)]()

This repository contains our code to answer the Machine Learning class Homework 2 No 9. We will perform a **DCGAN (Deep Convolutional Generative Adversarial Network)** to generate fake image data by training them against [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset

## Prerequisites

We use TensorFlow version 1.11 to run the experiment. So, you need to install TensorFlow before running this python file. We also use imageio to generate gif image. Thus, you need to install it as well.

```
pip install tensorflow=1.11.0
pip install imageio
```

## How to run

### Evaluating

You can start running the program **(WITHOUT THE NEED OF RE-TRAINING)** by evaluating the previously saved trained model using this following command. This command will also plot the history of the previous training result.

```
cd YOUR_DIR
# You need to specify one argument, which is the scenario that you want to run
# 8b, 8c, 8d, 8e, or 8f
python evaluate.py 8b
python evaluate.py 8c
python evaluate.py 8d
python evaluate.py 8e
python evaluate.py 8f
```

### Training

If you want to retrain the model, you can run this following command.

```
cd YOUR_DIR
python train_8b.py
python train_8c.py
python train_8d.py
python train_8e.py
python train_8f.py
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

## Scenario 1 - Using initial generator and discriminator model

What we mean by initial generator and discriminator model is 


The generator model configuration is like this

Layer Architecture:
* Dense -> BatchNorm -> ReLU
* Conv2DTranspose -> BatchNorm -> ReLU
* Conv2DTranspose -> BatchNorm -> ReLU
* Conv2DTranspose --> tanh

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

The discriminator model configuration is like this
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

## Scenario 2 - Using weaker generator model

The weak generator model configuration is like this
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              multiple                  313600
_________________________________________________________________
batch_normalization (BatchNo multiple                  12544
_________________________________________________________________
conv2d_transpose (Conv2DTran multiple                  12800
_________________________________________________________________
batch_normalization_1 (Batch multiple                  32
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr multiple                  200
_________________________________________________________________
dropout_1 (Dropout)          multiple                  0
=================================================================
Total params: 339,176
Trainable params: 332,888
Non-trainable params: 6,288
_________________________________________________________________
```

The discriminator model is like this
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
dense (Dense)                multiple                  6273
=================================================================
Total params: 212,865
Trainable params: 212,865
Non-trainable params: 0
_________________________________________________________________

```

## Scenario 3 - Using weaker discriminator model

The generator model configuration is like this
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

The weak discriminator model configuration is like this
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

## Authors

**Oktian Yustus Eko** - *Initial Work* - [mrkazawa](https://github.com/mrkazawa)