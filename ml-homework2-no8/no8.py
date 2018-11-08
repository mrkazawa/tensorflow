import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Data Information
IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
VALIDATION_SIZE = 5000

RANDOM_STATE = 2018
#Model
NO_EPOCHS = 10
BATCH_SIZE = 128

class Runner:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def load_fashion_mnist_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()

    def centering_the_image(self, x_train, x_test):
        x_train_mean = np.mean(x_train)
        x_train_stdev = np.std(x_train)
        return x_train - x_train_mean / x_train_stdev, x_test - x_train_mean / x_train_stdev

    def make_image_has_the_same_scale(self, x_train, x_test):
        return x_train.astype('float32') / 255, x_test.astype('float32') / 255

    def reshaping_image(self, x_train, x_test):
        return x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

    def perform_one_hot_encoding(self, y_train, y_test):
        return tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    def split_validation_data_from_train_data(self, x_train, y_train):
        (x_train, x_valid) = x_train[VALIDATION_SIZE:], x_train[:VALIDATION_SIZE]
        (y_train, y_valid) = y_train[VALIDATION_SIZE:], y_train[:VALIDATION_SIZE]
        return x_train, y_train, x_valid, y_valid

    def normalize_data(self):
        """Centering and Normalization

        Perform centering by mean subtraction, and normalization by dividing with 
        the standard deviation of the training dataset.

        Args:
        x_train (numpy.ndarray): x train data from fmnist
        x_test (numpy.ndarray): x test data from fmnist

        Returns:
        x_train, x_test: x train and x test normalized data
        """
        self.x_train, self.x_test = self.centering_the_image(self.x_train, self.x_test)
        self.x_train, self.x_test = self.make_image_has_the_same_scale(self.x_train, self.x_test)
        self.x_train, self.x_test = self.reshaping_image(self.x_train, self.x_test)
        self.y_train, self.y_test = self.perform_one_hot_encoding(self.y_train, self.y_test)
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.split_validation_data_from_train_data(self.x_train, self.y_train)
    
    def print_data_normalization_result(self):
        print("Fashion MNIST Normalization Result:")
        print("Training set (images) shape: {shape}".format(shape=self.x_train.shape))
        print("Training set (labels) shape: {shape}".format(shape=self.y_train.shape))
        print("Validation set (images) shape: {shape}".format(shape=self.x_valid.shape))
        print("Validation set (labels) shape: {shape}".format(shape=self.y_valid.shape))
        print("Test set (images) shape: {shape}".format(shape=self.x_test.shape))
        print("Test set (labels) shape: {shape}".format(shape=self.y_test.shape))

    def construct_model(self, initialization='he_normal'):
        # Add convolution 2D
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        kernel_initializer=initialization,
                        input_shape=(IMG_ROWS, IMG_COLS, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, 
                        kernel_size=(3, 3), 
                        activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    def run_training(self, n_epoch):
        self.train_model = self.model.fit(self.x_train, self.y_train,
                  batch_size=BATCH_SIZE,
                  epochs=n_epoch,
                  verbose=1,
                  validation_data=(self.x_valid, self.y_valid))

    def run_prediction(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def plot_accuracy_and_loss(self, title):
        hist = self.train_model.history
        acc = hist['acc']
        val_acc = hist['val_acc']
        loss = hist['loss']
        val_loss = hist['val_loss']
        epochs = range(len(acc))

        f, ax = plt.subplots(1,2, figsize=(14,6))
        f.canvas.set_window_title(title)

        ax[0].set_title(title + '\nTraining and validation accuracy')
        ax[0].set_ylim([0.7,1])
        ax[0].plot(epochs, acc, '--g', label='Training accuracy')
        ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
        ax[0].legend()

        ax[1].set_title(title + '\nTraining and validation loss')
        ax[1].set_ylim([0,0.6])
        ax[1].plot(epochs, loss, '--g', label='Training loss')
        ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
        ax[1].legend()
        plt.show(block=False)

def run_8b_he():
    runner = Runner()
    runner.load_fashion_mnist_data()
    runner.normalize_data()
    runner.print_data_normalization_result()
    runner.construct_model()
    runner.run_training(10)
    runner.run_prediction()
    runner.plot_accuracy_and_loss("8b - He Initialization")

def run_8b_xavier():
    runner = Runner()
    runner.load_fashion_mnist_data()
    runner.normalize_data()
    runner.print_data_normalization_result()
    runner.construct_model(tf.contrib.layers.xavier_initializer())
    runner.run_training(10)
    runner.run_prediction()
    runner.plot_accuracy_and_loss("8b - Xavier Initialization")

run_8b_he()
run_8b_xavier()

plt.show()