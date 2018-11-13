import tensorflow as tf
import utils

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected
def build_model(kernel_initializer):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (2, 2),
            strides = (1, 1),
            kernel_initializer = kernel_initializer,
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

n_epoch = 15
batch_size = 128

def run_scenario(kernel_initializer):
    runner.print_separator(kernel_initializer)
    model = None

    if (kernel_initializer == 'he_normal'):
        model = build_model(kernel_initializer = tf.keras.initializers.he_normal())
    elif (kernel_initializer == 'he_uniform'):
        model = build_model(kernel_initializer = tf.keras.initializers.he_uniform())
    elif (kernel_initializer == 'xavier_normal'):
        model = build_model(kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    elif (kernel_initializer == 'xavier_uniform'):
        model = build_model(kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = True))
    
    t_result, t_time = runner.start_training(model, n_epoch, batch_size)
    p_result = runner.start_prediction(model)

    training_result_list.append(t_result)
    training_time_list.append(t_time)
    training_title_list.append(kernel_initializer)
    prediction_result_list.append(p_result)

run_scenario('he_normal')
run_scenario('he_uniform')
run_scenario('xavier_normal')
run_scenario('xavier_uniform')

runner.print_result(training_title_list, training_time_list, prediction_result_list)
runner.plot_accuracy_and_loss(training_title_list, training_result_list)