import tensorflow as tf
import json
import utils


N_EPOCH = 30
BATCH_SIZE = 128

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> (Dropout) -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected


def build_model(activation):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation=activation,
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation=activation))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation=activation))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10,
                                    kernel_initializer='he_normal',
                                    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model


def build_prelu_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='linear',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='linear'))
    model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='linear'))
    model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10,
                                    kernel_initializer='he_normal',
                                    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model


def build_leaky_relu_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='linear',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='linear'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='linear'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10,
                                    kernel_initializer='he_normal',
                                    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model


def train_scenario(activation):
    runner.print_separator(activation)
    model = None

    if (activation == 'relu'):
        model = build_model(activation='relu')
    elif (activation == 'selu'):
        model = build_model(activation='selu')
    elif (activation == 'prelu'):
        model = build_prelu_model()
    elif (activation == 'leakyrelu'):
        model = build_leaky_relu_model()

    trained_model, training_history, training_time = runner.start_training(
        model, N_EPOCH, BATCH_SIZE)
    print('training time: %s' % (training_time))

    # Saving
    model_file_name = 'model_8e_%s.h5' % (activation)
    runner.save_trained_model_in_hdf5(trained_model, model_file_name)
    history_file_name = 'history_8e_%s.json' % (activation)
    runner.save_training_history_in_json(training_history, history_file_name)


runner = utils.Runner()
runner.load_fashion_mnist_data()
runner.normalize_data()
runner.print_data_normalization_result()

train_scenario('relu')
train_scenario('selu')
train_scenario('prelu')
train_scenario('leakyrelu')
