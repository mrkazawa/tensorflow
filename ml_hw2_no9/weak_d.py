import tensorflow as tf
import utils

EPOCHS = 300
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

tf.enable_eager_execution()


class WeakDiscriminator(tf.keras.Model):
    def __init__(self):
        super(WeakDiscriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            16, kernel_size=(5, 5), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(
            32, kernel_size=(5, 5), padding='same')
        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = self.pooling(x)
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.pooling(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def main():
    # cleaning
    utils.remove_all_files_inside_folder('./results/')
    utils.remove_all_files_inside_folder('./training_checkpoints/')
    # prepare dataset
    (train_images, _), (_, _) = utils.get_fmnist_data()
    train_dataset = utils.normalize_data(train_images)
    # create models
    generator = utils.Generator()
    discriminator = WeakDiscriminator()
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