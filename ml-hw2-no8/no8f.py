import tensorflow as tf
import utils

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


runner = utils.Runner()
runner.load_fashion_mnist_data()
runner.normalize_data()
runner.print_data_normalization_result()

training_result_list = list()
training_time_list = list()
training_title_list = list()
prediction_result_list = list()

n_epoch = 25
batch_size = 128


def run_scenario(regularizer):
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

    t_result, t_time = runner.start_training(model, n_epoch, batch_size)
    p_result = runner.start_prediction(model)

    training_result_list.append(t_result)
    training_time_list.append(t_time)
    training_title_list.append(regularizer)
    prediction_result_list.append(p_result)


run_scenario('none')
run_scenario('dropout')
run_scenario('l1')
run_scenario('l2')

runner.print_result(training_title_list, training_time_list,
                    prediction_result_list)
runner.plot_accuracy_and_loss(training_title_list, training_result_list)
