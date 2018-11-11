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
