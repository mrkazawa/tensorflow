import tensorflow as tf
import utils

# Popular CNN layer model is
# Conv -> ReLU -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> (Dropout) --> Fully Connected
def build_ideal_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (2, 2),
            strides = (1, 1),
            kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
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

def build_underfitting_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (3, 3),
            strides = (1, 1),
            kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
            activation = 'relu',
            input_shape = (28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (3, 3),  activation = 'relu'))
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

def build_overfitting_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32,
            kernel_size = (2, 2),
            strides = (1, 1),
            kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
            activation = 'relu',
            input_shape = (28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters = 64,
            kernel_size = (2, 2),  activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(filters = 128,
            kernel_size = (2, 2),  activation = 'relu'))
    model.add(tf.keras.layers.Conv2D(filters = 256,
            kernel_size = (2, 2),  activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(filters = 512,
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

    if (kernel_initializer == '17,578 params'):
        model = build_underfitting_model()
    elif (kernel_initializer == '87,402 params'):
        model = build_ideal_model()
    elif (kernel_initializer == '717,930 params'):
        model = build_overfitting_model()
    
    t_result, t_time = runner.start_training(model, n_epoch, batch_size)
    p_result = runner.start_prediction(model)

    training_result_list.append(t_result)
    training_time_list.append(t_time)
    training_title_list.append(kernel_initializer)
    prediction_result_list.append(p_result)

run_scenario('17,578 params')
run_scenario('87,402 params')
run_scenario('717,930 params')

runner.print_result(training_title_list, training_time_list, prediction_result_list)
runner.plot_accuracy_and_loss(training_title_list, training_result_list)