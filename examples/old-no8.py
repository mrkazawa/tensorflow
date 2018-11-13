import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Data Information
IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
VALIDATION_SIZE = 5000

#Model
NO_EPOCHS = 10
BATCH_SIZE = 128

class Runner:
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

    def start_training(self, model, n_epoch, batch_size):
        start_time = time.time()
        train_model = model.fit(self.x_train, self.y_train,
                  batch_size = batch_size,
                  epochs = n_epoch,
                  verbose = 0,
                  validation_data = (self.x_valid, self.y_valid))
        train_time = round(time.time() - start_time, 2)
        print("Training time: %s seconds ---" % (train_time))
        return train_model, train_time

    def start_prediction(self, model):
        score = model.evaluate(self.x_test, self.y_test, verbose = 0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score


class CNNModelBuilder:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.initilization = 'he_normal'
        self.activation = 'relu'
        self.n_hidden_layer = 2
        self.n_node = 32
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.input_shape = (IMG_ROWS, IMG_COLS, 1)
        self.is_regulated = False
        self.kernel_regularizer = None
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'

    def add_input_layer(self):
        self.model.add(tf.keras.layers.Conv2D(self.n_node,
                kernel_size = self.kernel_size,
                activation = self.activation,
                kernel_initializer = self.initilization,
                input_shape = self.input_shape))
    
    def add_hidden_layer(self):
        for i in range(self.n_hidden_layer):
            self.n_node = self.n_node * 2
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size = self.pool_size))
            if (self.is_regulated): self.model.add(tf.keras.layers.Dropout(0.25))
            self.model.add(tf.keras.layers.Conv2D(self.n_node,
                    kernel_size = self.kernel_size,
                    activation = self.activation,
                    kernel_regularizer = self.kernel_regularizer))

    def add_output_layer(self):
        if (self.is_regulated): self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.n_node,
                activation = self.activation,
                kernel_regularizer = self.kernel_regularizer))
        if (self.is_regulated): self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))

    def add_optimizer(self):
        self.model.compile(loss = self.loss,
                optimizer = self.optimizer,
                metrics = ['accuracy'])

    def construct(self):
        self.add_input_layer()
        self.add_hidden_layer()
        self.add_output_layer()
        self.add_optimizer()
        self.model.summary()
        return self.model


#plt.show()
def tabulate():
    print(10)

def print_banner(title):
    print('\n---------- ' + title + ' ----------')

runner = Runner()
runner.load_fashion_mnist_data()
runner.normalize_data()
runner.print_data_normalization_result()

scenario_8b_train_model = list()
scenario_8b_train_time = list()

def run_scenario_8b(initializer):
    print_banner('Scenario 8b %s Initialization' % (initializer))
    cnnBuilder = CNNModelBuilder()
    if (initializer == 'Xavier'): cnnBuilder.initilization = tf.contrib.layers.xavier_initializer()
    if (initializer == 'He'): cnnBuilder.initilization = 'he_normal'
    model = cnnBuilder.construct()
    runner.start_training(model, 10, 128)
    runner.start_prediction(model)

#run_scenario_8b(initializer = 'He')
#run_scenario_8b(initializer = 'Xavier')

# BEWARE! Running too many node will eats your memory. To mitigate: REDUCE YOUR BATCH_SIZE
# BEWARE! Adding more layer may made the pooling layer to reach negative. To mitigate: REDUCE THE POOL_SIZE
def run_scenario_8c(n_node, n_hidden_layer):
    cnnBuilder = CNNModelBuilder()
    cnnBuilder.n_node = n_node
    cnnBuilder.n_hidden_layer = n_hidden_layer
    print_banner("Scenario 8c n_node:%s n_hidden_layer:%s" % (n_node, n_hidden_layer))
    model = cnnBuilder.construct()
    runner.start_training(model, 10, 128)
    runner.start_prediction(model)
"""
run_scenario_8c(n_node = 32, n_hidden_layer = 0)
run_scenario_8c(n_node = 32, n_hidden_layer = 2)
run_scenario_8c(n_node = 16, n_hidden_layer = 0)
run_scenario_8c(n_node = 16, n_hidden_layer = 2)
"""
def run_scenario_8d(optimizer):
    print_banner('Scenario 8d %s Optimizer' % (optimizer))
    cnnBuilder = CNNModelBuilder()
    if (optimizer == 'Momentum'): cnnBuilder.optimizer = tf.train.MomentumOptimizer(learning_rate = 0.2, momentum = 0.1)
    elif (optimizer == 'Adam'): cnnBuilder.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999)
    model = cnnBuilder.construct()
    runner.start_training(model, 10, 128)
    runner.start_prediction(model)

#run_scenario_8d(optimizer = 'Adam')
#run_scenario_8d(optimizer = 'Momentum')

def run_scenario_8e(activation):
    print_banner('Scenario 8e %s Activation' % (activation))
    cnnBuilder = CNNModelBuilder()
    if (activation == 'Selu'): cnnBuilder.activation = 'selu'
    elif (activation == 'LeakyRelu'): cnnBuilder.activation = tf.keras.layers.LeakyReLU()
    elif (activation == 'Relu'): cnnBuilder.activation = 'relu'
    model = cnnBuilder.construct()
    runner.start_training(model, 10, 128)
    runner.start_prediction(model)

"""
run_scenario_8e(activation = 'Relu')
run_scenario_8e(activation = 'Selu')
run_scenario_8e(activation = 'LeakyRelu')
"""

def run_scenario_8f(regulation):
    print_banner('Scenario 8b %s Regulation' % (regulation))
    cnnBuilder = CNNModelBuilder()
    if (regulation == 'Dropout'): cnnBuilder.is_regulated = True
    elif (regulation == 'L2'): cnnBuilder.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.01)
    elif (regulation == 'L1'): cnnBuilder.kernel_regularizer = tf.contrib.layers.l1_regularizer(scale = 0.0001)
    model = cnnBuilder.construct()
    runner.start_training(model, 50, 128)
    runner.start_prediction(model)

#run_scenario_8f("None")
#run_scenario_8f("Dropout")
#run_scenario_8f("L2")
run_scenario_8f("L1")