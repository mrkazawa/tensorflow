# Machine Learning HW 2 - No 8

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()
[![dependency](https://img.shields.io/badge/dependency-tensorflow-orange.svg)]()

This repository contains our code to answer the Machine Learning class Homework 2 No 8. We will perform a CNN to classify the image data provided by the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset

## Prerequisites

We use TensorFlow version 1.11 to run the experiment. So, you need to install TensorFlow before running this python file

```
pip install tensorflow=1.11.0
```

## How to run

### Evaluating

You can start running the program **(WITHOUT THE NEED OF RE-TRAINING)** by evaluating the previously saved trained model using this following command. This command will also plot the history of the previous training result.

```shell
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

```shell
cd YOUR_DIR
python train_8b.py
python train_8c.py
python train_8d.py
python train_8e.py
python train_8f.py
```

## Default Model Architecture

```
Conv -> ReLU -> Conv -> ReLU -> Pool -> (Dropout) -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected
```

We will use the model structured above as our default Model to run the experiment. Otherwise, if we change the architecture of the model, we will state the changes in the corresponding pragraph.

## Scenario 8a - Preparing The Dataset

The first step of the algorithm is obviously to insert the dataset into our code. Then, we continue to perform data normalization before we begin the training. We will normalize both image data (x_axis) and image labels (y_axis). The normalization process can be divided into several steps as follows.

### 1 - Scaling the image

Ideally a Convolutional Neural Network will converge despite taking 0 – 255 as inputs (original greyscale color bit) instead of scaled down to 0 - 1. However, it will converge very slowly. Thus, we need to scale them with this code.

```python
def make_image_has_the_same_scale(self, x_train, x_test):
    return x_train.astype('float32') / 255, x_test.astype('float32') / 255
```

### 2 - Centering the image

Subtracting the dataset mean serves to "center" the data. Additionally, we ideally would like to divide by the standrad deviation of that feature or pixel as well if we want to normalize each feature value to a z-score = 0.

```python
def centering_the_image(self, x_train, x_test):
    x_train_mean = np.mean(x_train)
    x_train_stdev = np.std(x_train)
    return x_train - x_train_mean / x_train_stdev, x_test - x_train_mean / x_train_stdev
```

### 3 - Reshaping the dimension of the image data

Transforming the image dataset to have 3D dimension to be trained by the convolution layer. This particular dimension is important because it needs to be compatible with the Model that we are trying to built. Otherwise, the program will error. Original image is in 2D 28 by 28 format, we add 1 into the 'depth' because it is greyscale image. Thus the new dimension will be 3D (28, 28, 1) As an additional information, colored image will have the 'depth' of 3 becuase it contains RGB value.

```python
def reshaping_image(self, x_train, x_test):
    return x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(x_test.shape[0], 28, 28, 1)
```

### 4 - Performing one hot encoding

For the data labels (Y data), we only need to add number of classess (10 types of fashion) into the labels. After the training, the data labels will have 10 additional information that is the probability of the classification based on the number of class in the dataset.

```python
def perform_one_hot_encoding(self, y_train, y_test):
    return tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
```

### 5 - Split between the train and validation dataset

We need to divide our training dataset and put some into validation dataset. The validation dataset will be useful to optimize our model during the training. In this code, we take 5000 data from the train data and put them to the valid data.

```python
def split_validation_data_from_train_data(self, x_train, y_train):
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]
    return x_train, y_train, x_valid, y_valid
```

## Scenario 8b - Choosing the initializer

In this scenario, we choose between using the 'He' initilzaiton or 'Xavier' initialization. He is similar to Xavier initialization, with the factor multiplied by two. In this method, the weights are initialized keeping in mind the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently. The weights are still random but differ in range depending on the size of the previous layer of neurons. This provides a controlled initialisation hence the faster and more efficient gradient descent.

Below is the result of our training:
![Result No 8b](img/8b.png?raw=true "8b")

And below is the result of the predicition test
```
---------- he_normal (BEST) ----------
prediction acc: 0.9302

---------- he_uniform ----------
prediction acc: 0.9283

---------- xavier_normal ----------
prediction acc: 0.928

---------- xavier_uniform ----------
prediction acc: 0.9304
```

All of the initializers work really well in training, this can be seen with the prediciton results being in a very close gap among all of them. However, the he_unifrom will perform slightly below the others during the validation. Generally, the initializer is going to depend on the activation function that you user. He initialization works better for layers with ReLu activation. Meanwhile, Xavier initialization works better for layers with sigmoid activation. Since we are using relu in this scenario, we devote he_normal to be the best result.

## Scenario 8c - Choosing the number of nodes and layers

In this scenario, we try to tweak our neural network configuration into several models. Basically we want to change the number of nodes and the number of layers in our network. This will result in different number of trainable parameters that are going to be feed into the training process. The higher the number of parameters means that the network can train with more information. Thus, generating more accurate predictions. In summary, we came up with three different network configurations.

The first configuration (underfitting model):
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 32)        9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 32)          0
_________________________________________________________________
dropout (Dropout)            (None, 5, 5, 32)          0
_________________________________________________________________
flatten (Flatten)            (None, 800)               0
_________________________________________________________________
dense (Dense)                (None, 10)                8010
_________________________________________________________________
softmax (Softmax)            (None, 10)                0
=================================================================
Total params: 17,578
Trainable params: 17,578
Non-trainable params: 0
_________________________________________________________________
```

The second configuration (ideal model):
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_2 (Conv2D)            (None, 27, 27, 32)        160
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 26, 26, 64)        8256
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 12, 12, 128)       32896
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 128)         0
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 6, 128)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                46090
_________________________________________________________________
softmax_1 (Softmax)          (None, 10)                0
=================================================================
Total params: 87,402
Trainable params: 87,402
Non-trainable params: 0
_________________________________________________________________
```

The third configuration (overfitting model):
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_5 (Conv2D)            (None, 27, 27, 32)        160
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 26, 26, 64)        8256
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 13, 13, 64)        0
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 12, 12, 128)       32896
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 11, 11, 256)       131328
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 5, 5, 256)         0
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 4, 4, 512)         524800
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 2, 2, 512)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 2, 2, 512)         0
_________________________________________________________________
flatten_2 (Flatten)          (None, 2048)              0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                20490
_________________________________________________________________
softmax_2 (Softmax)          (None, 10)                0
=================================================================
Total params: 717,930
Trainable params: 717,930
Non-trainable params: 0
_________________________________________________________________
```

Below is the result of our training:
![Result No 8c](img/8c.png?raw=true "8c")

And below is the result of the predicition test
```
---------- 17,578 params ----------
training time: 53.89
prediction acc: 0.9019

---------- 87,402 params (BEST) ----------
training time: 118.05
prediction acc: 0.9278

---------- 717,930 params ----------
training time: 177.03
prediction acc: 0.9159
```

Based on the number of trainable params only, we can see that the third model will consume more time in training becuase of higher number of params, which is obvious observation. However, when we see the prediction and validation results, we can observe the behaviour of the underfitting and overfitting model. In the first model, we train with small number of params. The accuracy is good enough but this model is underfitted model, becuase it cannot reach the optimum accuracy. The third model is overfitted becuase this model cannot generate significant result (compared to the ideal model) even though it is trained with a huge amount of params. The validation loss also start to raise after it converged which is an indication of overfitted model. The best solution is to find the right balance between the trainable params and the accuracy, which is presented in the second model in this scenario.

## Scenario 8d - Choosing the optimizer

In this scenario, we want to measure the impact of optimizer to the neural networks. There are four optimizers that are on the list: adam, adagrad, rmsprop, and adadelta. Optimization algorithms helps us to minimize (or maximize) an Objective function (another name for Error function) E(x) which is simply a mathematical function dependent on the Model’s internal learnable parameters which are used in computing the target values(Y) from the set of predictors(X) used in the model. For example — we call the Weights(W) and the Bias(b) values of the neural network as its internal learnable parameters which are used in computing the output values and are learned and updated in the direction of optimal solution i.e minimizing the Loss by the network’s training process and also play a major role in the training process of the Neural Network Model.

Below is the result of our training:
![Result No 8d](img/8d.png?raw=true "8d")

And below is the result of the predicition test
```
---------- adam (BEST) ----------
prediction acc: 0.9281

---------- adagrad ----------
prediction acc: 0.9127

---------- rmsprop ----------
prediction acc: 0.9247

---------- adadelta ----------
prediction acc: 0.9255
```

If we only consider the prediciton results, we can say that all of the optimizers are good enough since their accuracy gap is really small from one another. However, when we see the validation data result, we can clearly see that Adam optimizers outperfoms all of the other optimizers. Adam can reach convergence quicker than the other and also generate less amount of loss. For sparse data sets we should use one of the adaptive learning-rate methods. An additional benefit is that we do not need to adjust the learning rate but likely achieve the best results with the default value. Like what we did in this experiment, we did not modify the learning rate parameter from the optimizers.

## Scenario 8e - Choosing the activation

In this scenario, we want to try multiple activation functions in our neural network. The activation functions on the list are: relu, selu, prelu, and leakyrelu. We want to test them fairly. So, we test them in the same architecture model, which is the same number of nodes, the same initializer, the same optimizer, and so on. The only different is only the activation function.

Below is the result of our training:
![Result No 8e](img/8e.png?raw=true "8e")

And below is the result of the predicition test
```
---------- relu (BEST) ----------
training time: 100.01
prediction acc: 0.9249

---------- selu ----------
training time: 157.26
prediction acc: 0.9105

---------- prelu ----------
training time: 168.94
prediction acc: 0.9229

---------- leakyrelu ----------
training time: 150.64
prediction acc: 0.9237
```

From the training result we can see that selu is performing worst. We can also argue that relu is on a par with prelu because of the almost similar accuracy and loss. However, we can still say that relu is the best activation function becuase relu can complete the training faster in just 100.01 seconds, compared to prelu in 168.94 seconds. This value indicates that relu is an efficient activation functions.

## Scenario 8f - Choosing the regularizer

In this scenario we try to use three different regularizers: Dropout, L1, and L2. The purpose of the regularizer is to overcome the overfitting that most likely to happen when you train without any regularizer. Dropout have the rate parameter that will control how many nodes that active during the training process. Thus, it can slow down the training and reduce the chance of overfitting. The bigger number of rate means that the more nodes will be set inactive during the training.

L1 and L2 regularizations have a λ parameter which is directly proportional to the penalty: the larger λ the stronger penalty to find complex models and it will be more likely that the model will avoid them. Likewise, if λ is zero, regularization is deactivated. During our implementation, the λ for L1 need to be assigned very small (compared to L2) in order to produce a 'good' result. Thus, L1 is may not be suitable for this scenario. Below is our configuration for the Dropout, L1, and L2 regularizer.

```
tf.keras.layers.Dropout(rate = 0.3)
tf.keras.regularizers.l1(l = 0.0001)
tf.keras.regularizers.l2(l = 0.0001)
```

Below is the result of our training:
![Result No 8f](img/8f.png?raw=true "8f")

And below is the result of the predicition test
```
---------- none ----------
prediction acc: 0.9121

---------- dropout (BEST) ----------
prediction acc: 0.9238

---------- l1 ----------
prediction acc: 0.8273

---------- l2 ----------
prediction acc: 0.8739
```

From the validation loss chart, we can see that the training without any regularization (none) will become overfitted. The L1 suffer from the underfitting scenario because we put poor lambda value (l=0.0001). If we change this lambda with more smaller number (e.g. l=0.0000001), it will produce better result. However, smaller number means that smaller impact the L1 given to the training data. L2 somehow perform better than L1 when using the same lambda value. Finally, Dropout is the best solution to overcome the overfitting problems.

## Authors

**Oktian Yustus Eko** - *Initial Work* - [mrkazawa](https://github.com/mrkazawa)