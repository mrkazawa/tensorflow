import tensorflow as tf
import json
import utils


N_EPOCH = 30
BATCH_SIZE = 128

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> (Dropout) -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected


def build_model(regularizer):
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
    # We put our regularizer here, after the output of max pooling
    if (regularizer != 'none' and regularizer != 'dropout'):
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), activity_regularizer=regularizer))
    if (regularizer == 'dropout'):
        model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(2, 2),
                                     kernel_initializer='he_normal',
                                     activation='relu'))
    # We put our regularizer here, after the output of max pooling
    if (regularizer != 'none' and regularizer != 'dropout'):
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), activity_regularizer=regularizer))
    if (regularizer == 'dropout'):
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


def train_scenario(regularizer):
    runner.print_separator(regularizer)
    model = None

    if (regularizer == 'none'):
        model = build_model(regularizer='none')
    if (regularizer == 'dropout'):
        model = build_model(regularizer='dropout')
    elif (regularizer == 'l1'):
        model = build_model(regularizer=tf.keras.regularizers.l1(l=0.0001))
    elif (regularizer == 'l2'):
        model = build_model(regularizer=tf.keras.regularizers.l2(l=0.0001))

    trained_model, training_history, training_time = runner.start_training(
        model, N_EPOCH, BATCH_SIZE)
    print('training time: %s' % (training_time))

    # Saving
    model_file_name = 'model_8f_%s.h5' % (regularizer)
    runner.save_trained_model_in_hdf5(trained_model, model_file_name)
    history_file_name = 'history_8f_%s.json' % (regularizer)
    runner.save_training_history_in_json(training_history, history_file_name)


runner = utils.Runner()
runner.load_fashion_mnist_data()
runner.normalize_data()
runner.print_data_normalization_result()

train_scenario('none')
train_scenario('dropout')
train_scenario('l1')
train_scenario('l2')
