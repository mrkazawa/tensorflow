import tensorflow as tf
import utils

EPOCHS = 300
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

tf.enable_eager_execution()


class WeakGenerator(tf.keras.Model):
    def __init__(self):
        super(WeakGenerator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(7*7*16, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            8, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x)

        x = tf.reshape(x, shape=(-1, 7, 7, 16))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x)

        x = tf.nn.tanh(self.conv2(x))
        return x


def main():
    # cleaning
    utils.remove_all_files_inside_folder('./results/')
    utils.remove_all_files_inside_folder('./training_checkpoints/')
    # prepare dataset
    (train_images, _), (_, _) = utils.get_fmnist_data()
    train_dataset = utils.normalize_data(train_images)
    # create models
    generator = WeakGenerator()
    discriminator = utils.Discriminator()
    # Defun gives 10 secs/epoch performance boost
    generator.call = tf.contrib.eager.defun(generator.call)
    discriminator.call = tf.contrib.eager.defun(discriminator.call)
    # training helpers
    checkpoint = utils.setup_checkpoint(generator, discriminator)
    random_vector = utils.generate_constant_random_vector(
        NOISE_DIM, NUM_EXAMPLES_TO_GENERATE)
    # training
    history = utils.train(dataset=train_dataset, epochs=EPOCHS, noise_dim=NOISE_DIM, generator=generator,
                          discriminator=discriminator, checkpoint=checkpoint, random_vector=random_vector)
    # reporting
    generator.summary()
    discriminator.summary()
    utils.plot_loss(history)
    utils.create_gif()


if __name__ == "__main__":
    main()
