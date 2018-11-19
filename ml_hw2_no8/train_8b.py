import tensorflow as tf
import json
import utils


N_EPOCH = 30
BATCH_SIZE = 128

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> (Dropout) -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected


def build_model(kernel_initializer):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(2, 2),
                                     kernel_initializer=kernel_initializer,
                                     activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(2, 2),
                                     kernel_initializer=kernel_initializer,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(2, 2),
                                     kernel_initializer=kernel_initializer,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10,
                                    kernel_initializer=kernel_initializer,
                                    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model


def train_scenario(kernel_initializer):
    runner.print_separator(kernel_initializer)
    model = None

    if (kernel_initializer == 'he_normal'):
        model = build_model(
            kernel_initializer=tf.keras.initializers.he_normal())
    elif (kernel_initializer == 'he_uniform'):
        model = build_model(
            kernel_initializer=tf.keras.initializers.he_uniform())
    elif (kernel_initializer == 'xavier_normal'):
        model = build_model(
            kernel_initializer=tf.keras.initializers.glorot_normal())
    elif (kernel_initializer == 'xavier_uniform'):
        model = build_model(
            kernel_initializer=tf.glorot_uniform_initializer())

    trained_model, training_history, training_time = runner.start_training(
        model, N_EPOCH, BATCH_SIZE)
    print('training time: %s' % (training_time))

    # Saving
    model_file_name = 'model_8b_%s.h5' % (kernel_initializer)
    runner.save_trained_model_in_hdf5(trained_model, model_file_name)
    history_file_name = 'history_8b_%s.json' % (kernel_initializer)
    runner.save_training_history_in_json(training_history, history_file_name)


runner = utils.Runner()
runner.load_fashion_mnist_data()
runner.normalize_data()
runner.print_data_normalization_result()

train_scenario('he_normal')
train_scenario('he_uniform')
train_scenario('xavier_normal')
train_scenario('xavier_uniform')
