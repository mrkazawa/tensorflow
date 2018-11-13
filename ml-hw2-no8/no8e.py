import tensorflow as tf
import utils

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected
def build_model(activation):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (2, 2),
            strides = (1, 1),
            kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
            activation = activation,
            input_shape = (28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters = 64,
            kernel_size = (2, 2),  activation = activation))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(filters = 128,
            kernel_size = (2, 2),  activation = activation))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(rate = 0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 10))
    model.add(tf.keras.layers.Softmax())

    model.compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])
    
    model.summary()
    return model

def build_prelu_model():
    """ Build a model specifically for PReLU activation
    Somehow the PReLU cannot be combined with the other activation
    in the same build_model() method. A strange behaviour.
    """
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (2, 2),
            strides = (1, 1),
            kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
            activation = tf.keras.layers.PReLU(shared_axes=[1, 2]),
            input_shape = (28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters = 64,
            kernel_size = (2, 2),  activation = tf.keras.layers.PReLU(shared_axes=[1, 2])))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(filters = 128,
            kernel_size = (2, 2),  activation = tf.keras.layers.PReLU(shared_axes=[1, 2])))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(rate = 0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 10))
    model.add(tf.keras.layers.Softmax())

    model.compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
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

    if (optimizer == 'relu'):
        model = build_model(activation = 'relu')
    elif (optimizer == 'selu'):
        model = build_model(activation = 'selu')
    elif (optimizer == 'prelu'):
        model = build_prelu_model()
    elif (optimizer == 'leakyrelu'):
        model = build_model(activation = tf.keras.layers.LeakyReLU())
    
    t_result, t_time = runner.start_training(model, n_epoch, batch_size)
    p_result = runner.start_prediction(model)

    training_result_list.append(t_result)
    training_time_list.append(t_time)
    training_title_list.append(optimizer)
    prediction_result_list.append(p_result)

run_scenario('relu')
run_scenario('selu')
run_scenario('prelu')
run_scenario('leakyrelu')

runner.print_result(training_title_list, training_time_list, prediction_result_list)
runner.plot_accuracy_and_loss(training_title_list, training_result_list)