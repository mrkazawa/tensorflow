import tensorflow as tf
import json
import utils


N_EPOCH = 30
BATCH_SIZE = 128

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> (Dropout) -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected


def build_ideal_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
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


def build_underfitting_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
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


def build_overfitting_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
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


def train_scenario(layer):
    runner.print_separator(layer)
    model = None

    if (layer == 'underfit'):
        model = build_underfitting_model()
    elif (layer == 'ideal'):
        model = build_ideal_model()
    elif (layer == 'overfit'):
        model = build_overfitting_model()

    trained_model, training_history, training_time = runner.start_training(
        model, N_EPOCH, BATCH_SIZE)
    print('training time: %s' % (training_time))

    # Saving
    model_file_name = 'model_8c_%s.h5' % (layer)
    runner.save_trained_model_in_hdf5(trained_model, model_file_name)
    history_file_name = 'history_8c_%s.json' % (layer)
    runner.save_training_history_in_json(training_history, history_file_name)


runner = utils.Runner()
runner.load_fashion_mnist_data()
runner.normalize_data()
runner.print_data_normalization_result()

train_scenario('underfit')
train_scenario('ideal')
train_scenario('overfit')
