from IPython import display
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import imageio
import glob

BUFFER_SIZE = 60000
BATCH_SIZE = 256

PATH_TO_CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(PATH_TO_CHECKPOINT_DIR, "ckpt")

PATH_TO_RESULT_DIR = './results/'
IMAGE_FILE_NAME_TEMPLATE = 'image_at_epoch_{:04d}.png'

DISCRIMINATOR_OPTIMIZER = tf.train.AdamOptimizer(1e-4)
GENERATOR_OPTIMIZER = tf.train.AdamOptimizer(1e-4)

history = {}
history['g_losses'] = []
history['d_losses'] = []


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2DTranspose(
            32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)

        x = tf.nn.tanh(self.conv3(x))
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def remove_all_files_inside_folder(folder_dir):
    for the_file in os.listdir(folder_dir):
        file_path = os.path.join(folder_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_fmnist_data():
    return tf.keras.datasets.fashion_mnist.load_data()


def _reshape_images(train_images):
    # Reshape the image to have 3D dimension 28x28 is the image size
    # and 1 is the grayscale depth
    return train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')


def _centering_images(train_images):
    # We are normalizing the images to the range of [-1, 1]
    return (train_images - 127.5) / 127.5


def _create_custom_dataset(data):
    # Create custom dataset that only contains image data
    return tf.data.Dataset.from_tensor_slices(data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def normalize_data(train_images):
    train_images = _reshape_images(train_images)
    train_images = _centering_images(train_images)
    return _create_custom_dataset(data=train_images)


def _calculate_discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)
    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss
    return total_loss


def _calculate_generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def setup_checkpoint(generator, discriminator):
    return tf.train.Checkpoint(generator_optimizer=GENERATOR_OPTIMIZER,
                               discriminator_optimizer=DISCRIMINATOR_OPTIMIZER,
                               generator=generator,
                               discriminator=discriminator)


def generate_constant_random_vector(noise_dim, number_of_examples):
    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement of the gan.
    return tf.random_normal([number_of_examples, noise_dim])


def _generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # the generated image from predictions will be between -1 to 1
        # because of the previous normalization. Thus we need to change
        # them back to the grayscale bit between 0 to 255
        # reverse process from the centering_image() method
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(PATH_TO_RESULT_DIR+IMAGE_FILE_NAME_TEMPLATE.format(epoch))
    plt.close()


def train(dataset, epochs, noise_dim, generator, discriminator, checkpoint, random_vector):
    for epoch in range(epochs):
        start = time.time()
        g_losses = list()
        d_losses = list()

        for images in dataset:
            # generating noise from a uniform distribution
            noise = tf.random_normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                generated_output = discriminator(
                    generated_images, training=True)

                gen_loss = _calculate_generator_loss(generated_output)
                disc_loss = _calculate_discriminator_loss(
                    real_output, generated_output)

            g_losses.append(gen_loss.numpy())
            d_losses.append(disc_loss.numpy())

            gradients_of_generator = gen_tape.gradient(
                gen_loss, generator.variables)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, discriminator.variables)

            GENERATOR_OPTIMIZER.apply_gradients(
                zip(gradients_of_generator, generator.variables))
            DISCRIMINATOR_OPTIMIZER.apply_gradients(
                zip(gradients_of_discriminator, discriminator.variables))

        history["g_losses"].append(np.mean(g_losses))
        history["d_losses"].append(np.mean(d_losses))

        if epoch % 1 == 0:
            display.clear_output(wait=True)
            _generate_and_save_images(generator, epoch+1, random_vector)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

        print('Time taken for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))

    # generating after the final epoch
    display.clear_output(wait=True)
    _generate_and_save_images(generator, epochs, random_vector)

    return history


def create_gif():
    with imageio.get_writer(PATH_TO_RESULT_DIR+'dcgan.gif', mode='I') as writer:
        filenames = glob.glob(PATH_TO_RESULT_DIR+'image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def plot_loss(history):
    f, ax = plt.subplots(figsize=(14, 8))
    f.canvas.set_window_title('Result')

    ax.set_title('Training Loss')
    ax.set_xlabel('epoch')

    g_loss = history['g_losses']
    d_loss = history['d_losses']
    epochs = range(len(g_loss))

    ax.plot(epochs, g_loss, 'r', label='generator_loss')
    ax.plot(epochs, d_loss, 'g', label='discriminator_loss')

    ax.legend()
    plt.tight_layout()
    plt.savefig(PATH_TO_RESULT_DIR+'loss_chart.png')
    # plt.show()
