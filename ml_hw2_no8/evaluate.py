import tensorflow as tf
import sys
import utils

# -------------------- Scenario 8b --------------------

training_8b_labels = [
    'he_normal',
    'he_uniform',
    'xavier_normal',
    'xavier_uniform'
]

model_8b_file_names = [
    'model_8b_he_normal.h5',
    'model_8b_he_uniform.h5',
    'model_8b_xavier_normal.h5',
    'model_8b_xavier_uniform.h5'
]

training_8b_file_names = [
    'history_8b_he_normal.json',
    'history_8b_he_uniform.json',
    'history_8b_xavier_normal.json',
    'history_8b_xavier_uniform.json'
]

# -------------------- Scenario 8c --------------------

training_8c_labels = [
    '17,578 params',
    '87,402 params',
    '717,930 params'
]

model_8c_file_names = [
    'model_8c_underfit.h5',
    'model_8c_ideal.h5',
    'model_8c_overfit.h5'
]

training_8c_file_names = [
    'history_8c_underfit.json',
    'history_8c_ideal.json',
    'history_8c_overfit.json'
]

# -------------------- Scenario 8d --------------------

training_8d_labels = [
    'adam',
    'adagrad',
    'rmsprop',
    'adadelta'
]

model_8d_file_names = [
    'model_8d_adam.h5',
    'model_8d_adagrad.h5',
    'model_8d_rmsprop.h5',
    'model_8d_adadelta.h5'
]

training_8d_file_names = [
    'history_8d_adam.json',
    'history_8d_adagrad.json',
    'history_8d_rmsprop.json',
    'history_8d_adadelta.json'
]

# -------------------- Scenario 8e --------------------

training_8e_labels = [
    'relu',
    'selu',
    'prelu',
    'leakyrelu'
]

model_8e_file_names = [
    'model_8e_relu.h5',
    'model_8e_selu.h5',
    'model_8e_prelu.h5',
    'model_8e_leakyrelu.h5'
]

training_8e_file_names = [
    'history_8e_relu.json',
    'history_8e_selu.json',
    'history_8e_prelu.json',
    'history_8e_leakyrelu.json'
]

# -------------------- Scenario 8f --------------------

training_8f_labels = [
    'none',
    'dropout',
    'l1',
    'l2'
]

model_8f_file_names = [
    'model_8f_none.h5',
    'model_8f_dropout.h5',
    'model_8f_l1.h5',
    'model_8f_l2.h5'
]

training_8f_file_names = [
    'history_8f_none.json',
    'history_8f_dropout.json',
    'history_8f_l1.json',
    'history_8f_l2.json'
]


def evaluate(scenario, labels, model_file_names, history_file_names):
    for i in range(len(labels)):
        model = runner.load_model(model_file_names[i])
        score = runner.start_prediction(model)

        runner.print_separator(labels[i])
        print('accuracy: %s' % (score))

    runner.plot_accuracy_and_loss(scenario, labels, history_file_names)


def check_if_have_none_or_more_then_two_argument():
    return len(sys.argv) < 2 or len(sys.argv) > 2


def check_if_argument_value_correct():
    return sys.argv[1] != '8b' and sys.argv[1] != '8c' and sys.argv[1] != '8d' and sys.argv[1] != '8e' and sys.argv[1] != '8f'


def exit_and_print_error():
    sys.exit('You should specify one argument: (8b, 8c, 8d, 8e, or 8f)!')


if check_if_have_none_or_more_then_two_argument():
    exit_and_print_error()
elif check_if_argument_value_correct():
    exit_and_print_error()
else:
    scenario = sys.argv[1]

    runner = utils.Runner()
    runner.load_fashion_mnist_data()
    runner.normalize_data()
    runner.print_data_normalization_result()

    if (scenario == '8b'):
        evaluate(scenario, training_8b_labels,
                 model_8b_file_names, training_8b_file_names)
    elif (scenario == '8c'):
        evaluate(scenario, training_8c_labels,
                 model_8c_file_names, training_8c_file_names)
    elif (scenario == '8d'):
        evaluate(scenario, training_8d_labels,
                 model_8d_file_names, training_8d_file_names)
    elif (scenario == '8e'):
        evaluate(scenario, training_8e_labels,
                 model_8e_file_names, training_8e_file_names)
    elif (scenario == '8f'):
        evaluate(scenario, training_8f_labels,
                 model_8f_file_names, training_8f_file_names)
