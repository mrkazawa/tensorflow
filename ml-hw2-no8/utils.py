import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Data Information
IMG_PIXEL_ROWS = 28
IMG_PIXEL_COLS = 28
NUM_CLASSES = 10
PIXEL_LEVEL_RANGE = 255

VALIDATION_SIZE = 5000

class Runner:
    def load_fashion_mnist_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()

    def centering_the_image(self, x_train, x_test):
        """ Perform centering by mean subtraction and dividing with 
        the standard deviation of the training dataset. z-score = 0

        Args:
        x_train: train image data from dataset
        x_test: test image data from dataset

        Returns:
        x_train, x_test: train and test image data
        """
        x_train_mean = np.mean(x_train)
        x_train_stdev = np.std(x_train)
        return (x_train - x_train_mean) / x_train_stdev, (x_test - x_train_mean) / x_train_stdev

    def make_image_has_the_same_scale(self, x_train, x_test):
        """ Ideally a Convolutional Neural Network will converge despite taking 0–255
        as inputs instead of scaled down to 0-1. However, it will converge very slowly.

        Args:
        x_train: train image data from dataset
        x_test: test image data from dataset

        Returns:
        x_train, x_test: train and test image data
        """
        return x_train.astype('float32') / PIXEL_LEVEL_RANGE, x_test.astype('float32') / PIXEL_LEVEL_RANGE

    def reshaping_image(self, x_train, x_test):
        """ Transforming the image dataset to have 3D dimension to be trained by
        the convolution layer. Original image is in 28 by 28 format, we add 1 into the
        'depth' because it is greyscale image. Colored image will have the 'depth'
        of 3 becuase it contains RGB value.

        Args:
        x_train: train image data from dataset
        x_test: test image data from dataset

        Returns:
        x_train, x_test: train and test image data
        """
        return x_train.reshape(x_train.shape[0], IMG_PIXEL_ROWS, IMG_PIXEL_COLS, 1), x_test.reshape(x_test.shape[0], IMG_PIXEL_ROWS, IMG_PIXEL_COLS, 1)

    def perform_one_hot_encoding(self, y_train, y_test):
        """ Add the number of class information into the data labels.

        Args:
        x_train: train label data from dataset
        x_test: test label data from dataset

        Returns:
        x_train, x_test: train and test label data
        """
        return tf.keras.utils.to_categorical(y_train, NUM_CLASSES), tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    def split_validation_data_from_train_data(self, x_train, y_train):
        """ Divide the training data and put some of them into validation data.
        Validation data will be used to optimized the result of the training.

        Args:
        x_train: train image data from dataset
        x_test: test image data from dataset

        Returns:
        x_train, x_test: train and test image data
        """
        (x_train, x_valid) = x_train[VALIDATION_SIZE:], x_train[:VALIDATION_SIZE]
        (y_train, y_valid) = y_train[VALIDATION_SIZE:], y_train[:VALIDATION_SIZE]
        return x_train, y_train, x_valid, y_valid

    def normalize_data(self):
        self.x_train, self.x_test = self.make_image_has_the_same_scale(self.x_train, self.x_test)
        self.x_train, self.x_test = self.centering_the_image(self.x_train, self.x_test)
        self.x_train, self.x_test = self.reshaping_image(self.x_train, self.x_test)
        self.y_train, self.y_test = self.perform_one_hot_encoding(self.y_train, self.y_test)
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.split_validation_data_from_train_data(self.x_train, self.y_train)

    def start_training(self, model, n_epoch, batch_size):
        start_time = time.time()
        train_model = model.fit(self.x_train, self.y_train,
                  batch_size = batch_size,
                  epochs = n_epoch,
                  verbose = 0,
                  validation_data = (self.x_valid, self.y_valid))
        train_time = round(time.time() - start_time, 2)
        return train_model, train_time

    def start_prediction(self, model):
        score = model.evaluate(self.x_test, self.y_test, verbose = 0)
        return score

    def print_separator(self, title):
        print('\n---------- %s ----------'  % (title))

    def print_data_normalization_result(self):
        print("Fashion MNIST Normalization Result:")
        print("Training set (images) shape: {shape}".format(shape=self.x_train.shape))
        print("Training set (labels) shape: {shape}".format(shape=self.y_train.shape))
        print("Validation set (images) shape: {shape}".format(shape=self.x_valid.shape))
        print("Validation set (labels) shape: {shape}".format(shape=self.y_valid.shape))
        print("Test set (images) shape: {shape}".format(shape=self.x_test.shape))
        print("Test set (labels) shape: {shape}".format(shape=self.y_test.shape))

    def print_result(self, t_title_list, t_time_list, p_result_list):
        for i in range(len(t_title_list)):
            self.print_separator(t_title_list[i])
            print('training time: %s' % (t_time_list[i]))
            print('prediction loss: %s' % (p_result_list[i][0]))
            print('prediction acc: %s' % (p_result_list[i][1]))

    def plot_accuracy_and_loss(self, t_title_list, t_result_list):
        f, ax = plt.subplots(2, 2, figsize = (14, 6))
        f.canvas.set_window_title('Result')
        colors = ['b', 'g', 'r', 'm', 'c', 'y']

        ax[0, 0].set_title('Training accuracy')
        ax[0, 1].set_title('Validation accuracy')
        ax[1, 0].set_title('Training loss')
        ax[1, 1].set_title('Validation loss')

        for i in range(len(t_title_list)):
            hist = t_result_list[i].history
            acc = hist['acc']
            val_acc = hist['val_acc']
            loss = hist['loss']
            val_loss = hist['val_loss']
            epochs = range(len(acc))

            ax[0, 0].plot(epochs, acc, colors[i], label = t_title_list[i])
            ax[0, 1].plot(epochs, val_acc, colors[i], label = t_title_list[i])
            ax[1, 0].plot(epochs, loss, colors[i], label = t_title_list[i])
            ax[1, 1].plot(epochs, val_loss, colors[i], label = t_title_list[i])

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()

        plt.show()