import tensorflow as tf
import json
import utils


N_EPOCH = 30
BATCH_SIZE = 128

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> (Dropout) -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected


def build_model(optimizer):
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
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()
    return model


def train_scenario(optimizer):
    runner.print_separator(optimizer)
    model = None

    if (optimizer == 'adam'):
        model = build_model(optimizer=tf.keras.optimizers.Adam())
    elif (optimizer == 'adagrad'):
        model = build_model(optimizer=tf.keras.optimizers.Adagrad())
    elif (optimizer == 'rmsprop'):
        model = build_model(optimizer=tf.keras.optimizers.RMSprop())
    elif (optimizer == 'adadelta'):
        model = build_model(optimizer=tf.keras.optimizers.Adadelta())

    trained_model, training_history, training_time = runner.start_training(
        model, N_EPOCH, BATCH_SIZE)
    print('training time: %s' % (training_time))

    # Saving
    model_file_name = 'model_8d_%s.h5' % (optimizer)
    runner.save_trained_model_in_hdf5(trained_model, model_file_name)
    history_file_name = 'history_8d_%s.json' % (optimizer)
    runner.save_training_history_in_json(training_history, history_file_name)


runner = utils.Runner()
runner.load_fashion_mnist_data()
runner.normalize_data()
runner.print_data_normalization_result()

train_scenario('adam')
train_scenario('adagrad')
train_scenario('rmsprop')
train_scenario('adadelta')
