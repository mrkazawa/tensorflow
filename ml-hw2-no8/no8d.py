import tensorflow as tf
import utils

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected
def build_model(optimizer):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (2, 2),
            strides = (1, 1),
            kernel_initializer = 'he_normal',
            activation = 'relu',
            input_shape = (28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters = 64,
            kernel_size = (2, 2),  activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(filters = 128,
            kernel_size = (2, 2),  activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 10))
    model.add(tf.keras.layers.Softmax())

    model.compile(loss = 'categorical_crossentropy',
            optimizer = optimizer,
            metrics = ['accuracy'])
    
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

n_epoch = 15
batch_size = 128

def run_scenario(optimizer):
    runner.print_separator(optimizer)
    model = None

    if (optimizer == 'adam'):
        model = build_model(optimizer = tf.keras.optimizers.Adam())
    elif (optimizer == 'adagrad'):
        model = build_model(optimizer = tf.keras.optimizers.Adagrad())
    elif (optimizer == 'rmsprop'):
        model = build_model(optimizer = tf.keras.optimizers.RMSprop())
    elif (optimizer == 'adadelta'):
        model = build_model(optimizer = tf.keras.optimizers.Adadelta())
    
    t_result, t_time = runner.start_training(model, n_epoch, batch_size)
    p_result = runner.start_prediction(model)

    training_result_list.append(t_result)
    training_time_list.append(t_time)
    training_title_list.append(optimizer)
    prediction_result_list.append(p_result)

run_scenario('adam')
run_scenario('adagrad')
run_scenario('rmsprop')
run_scenario('adadelta')

runner.print_result(training_title_list, training_time_list, prediction_result_list)
runner.plot_accuracy_and_loss(training_title_list, training_result_list)