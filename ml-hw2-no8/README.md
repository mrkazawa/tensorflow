# Machinel Learning HW 2 - No 8

This repository contains our code to answer the Machine Learning class Homework 2 No 8. We will perform a CNN to classify the image data provided by the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset

## Prerequisites

We use TensorFlow version 1.11 to run the experiment. So, we need to install TensorFlow before running this python file

```
pip install tensorflow=1.11.0
```

## Default Model Architecture

```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 128)         73856
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0
_________________________________________________________________
dense (Dense)                (None, 128)               147584
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 241,546
Trainable params: 241,546
Non-trainable params: 0
```

We will use the Model above as our default Model to run the experiment. Otherwise, if we change the model, we will state the changes in the corresponding sub-section.

## Scenario 8a - Preparing The Dataset

The first step of the algorithm is obviously to insert the data into our code. Then, we continue to perform data normalization before we begin the training. We will normalize both image data (x_axis) and image labels (y_axis). The normalization process can be divided into several steps as follows.

### 1 - Centering the image

Subtracting the dataset mean serves to "center" the data. Additionally, we ideally would like to divide by the sttdev of that feature or pixel as well if we want to normalize each feature value to a z-score.

```
def centering_the_image(self, x_train, x_test):
    x_train_mean = np.mean(x_train)
    x_train_stdev = np.std(x_train)
    return x_train - x_train_mean / x_train_stdev, x_test - x_train_mean / x_train_stdev
```

### 2 - Scaling the image

We would like to change the image data into float and divide them by 255 to make all images to have the same scale.

```
def make_image_has_the_same_scale(self, x_train, x_test):
    return x_train.astype('float32') / 255, x_test.astype('float32') / 255
```

### 3 - Reshaping the dimension of the image data

We reshape all the image columns from (784) to (28,28,1). This particular dimension is important because it is compatible with the Model that we are trying to built. Otherwise, the program will error. IMG_ROWS = 28, IMG_COLS = 28

```
def reshaping_image(self, x_train, x_test):
    return x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
```

### 4 - Performing one hot encoding

For the data labels, we only need to add number of classess (10 types of fashion) into the labels. NUM_CLASSES = 10

```
def perform_one_hot_encoding(self, y_train, y_test):
    return tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
```

### 5 - Split between the train and validation dataset

We need to divide our training dataset and put some into validation dataset. The validation dataset will be useful to optimize our model during the training. VALIDATION_SIZE tells how many data that you want to put into the validation data.

```
def split_validation_data_from_train_data(self, x_train, y_train):
    (x_train, x_valid) = x_train[VALIDATION_SIZE:], x_train[:VALIDATION_SIZE]
    (y_train, y_valid) = y_train[VALIDATION_SIZE:], y_train[:VALIDATION_SIZE]
    return x_train, y_train, x_valid, y_valid
```


## Scenario 8b - Choosing the initializer

In this scenario, we must choose between using the 'He' initilzaiton or 'Xavier' initialization. We are going to implement both of them and analyze their behaviour during training and testing.

Below is the result of our implementation:
![Result No 8b](img/result-no8b.jpeg?raw=true "result-no8b")

And below is the result of the predicition test
```
---------- he_normal ----------
training time: 51.38
prediction loss: 0.2358957895487547
prediction acc: 0.9233

---------- he_uniform ----------
training time: 50.7
prediction loss: 0.2410036524027586
prediction acc: 0.9243

---------- xavier_normal ----------
training time: 50.09
prediction loss: 0.22017258661985398
prediction acc: 0.9244

---------- xavier_uniform ----------
training time: 50.96
prediction loss: 0.21316648482978343
prediction acc: 0.9302
```

## Scenario 8c - Choosing the number of nodes and layers



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

Below is the result of our implementation:
![Result No 8c](img/result-no8c.jpeg?raw=true "result-no8c")

And below is the result of the predicition test
```
---------- 17,578 params ----------
training time: 27.54
prediction loss: 0.2739635009288788
prediction acc: 0.9009

---------- 87,402 params ----------
training time: 47.43
prediction loss: 0.22836468553245068
prediction acc: 0.9219

---------- 717,930 params ----------
training time: 78.43
prediction loss: 0.3130791326530278
prediction acc: 0.9196
```

## Scenario 8d - Choosing the optimizer

Below is the result of our implementation:
![Result No 8d](img/result-no8d.jpeg?raw=true "result-no8d")

And below is the result of the predicition test
```
---------- adam ----------
training time: 51.49
prediction loss: 0.22701162923276424
prediction acc: 0.9235

---------- adagrad ----------
training time: 48.55
prediction loss: 0.228485681951046
prediction acc: 0.9194

---------- rmsprop ----------
training time: 49.6
prediction loss: 0.23045796354711057
prediction acc: 0.9219

---------- adadelta ----------
training time: 51.57
prediction loss: 0.22598704968690872
prediction acc: 0.9227
```

## Scenario 8e - Choosing the activation

Below is the result of our implementation:
![Result No 8e](img/result-no8e.jpeg?raw=true "result-no8e")

And below is the result of the predicition test
```
---------- relu ----------
training time: 52.57
prediction loss: 0.21136212078034877
prediction acc: 0.9299

---------- selu ----------
training time: 87.22
prediction loss: 0.27458372920155527
prediction acc: 0.9109

---------- prelu ----------
training time: 84.58
prediction loss: 0.2126349125891924
prediction acc: 0.9259

---------- leakyrelu ----------
training time: 79.83
prediction loss: 0.23446455146670342
prediction acc: 0.9191
```

## Scenario 8f - Choosing the regularizer

In this scenario we try to use three different regularizers: Dropout, L1, and L2. The purpose of the regularizer is to overcome the overfitting that most likely to happen when you train without any regularizer. Dropout have the rate parameter that will control how many nodes that active during the training process. Thus, it can slow down the training and reduce the chance of overfitting. The bigger number of rate means that the more nodes will be set inactive during the training.

L1 and L2 regularizations have a λ parameter which is directly proportional to the penalty: the larger λ the stronger penalty to find complex models and it will be more likely that the model will avoid them. Likewise, if λ is zero, regularization is deactivated. During our implementation, the λ for L1 need to be assigned very small (compared to L2) in order to produce a good result. Thus, L1 is not suitable for this scenario. Below is our configuration for the Dropout, L1, and L2 regularizer.

```
tf.keras.layers.Dropout(rate = 0.3)
tf.keras.regularizers.l1(l=0.0001)
tf.keras.regularizers.l2(l=0.001)
```

Below is the result of our implementation:
![Result No 8f](img/result-no8f.jpeg?raw=true "result-no8f")

And below is the result of the predicition test
```
---------- none ----------
training time: 49.68
prediction loss: 0.41594458510363475
prediction acc: 0.9141

---------- dropout ----------
training time: 54.97
prediction loss: 0.2601199430510402
prediction acc: 0.9268

---------- l1 ----------
training time: 50.5
prediction loss: 0.5417756761074066
prediction acc: 0.8208

---------- l2 ----------
training time: 49.78
prediction loss: 0.2600191069841385
prediction acc: 0.9179
```

Based on the training result

Based on the prediction result, we can see that the 

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
