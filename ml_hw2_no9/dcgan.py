import tensorflow as tf
import utils

EPOCHS = 300
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

tf.enable_eager_execution()


class StrongDiscriminator(tf.keras.Model):
    def __init__(self):
        super(StrongDiscriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            64, kernel_size=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(
            128, kernel_size=(2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(
            256, kernel_size=(2, 2), padding='same')
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv3(x))
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


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


class WeakGenerator(tf.keras.Model):
    def __init__(self):
        super(WeakGenerator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False)
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

        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x)

        x = tf.nn.tanh(self.conv2(x))
        return x


# cleaning
utils.remove_all_files_inside_folder('./results/')
utils.remove_all_files_inside_folder('./training_checkpoints/')

# prepare dataset
(train_images, train_labels), (_, _) = utils.get_fmnist_data()
train_dataset = utils.normalize_data(train_images)

# create models
#generator = utils.Generator()
discriminator = utils.Discriminator()

generator = WeakGenerator()
#discriminator = StrongDiscriminator()
#discriminator = WeakDiscriminator()


# Defun gives 10 secs/epoch performance boost
generator.call = tf.contrib.eager.defun(generator.call)
discriminator.call = tf.contrib.eager.defun(discriminator.call)



checkpoint = utils.setup_checkpoint(generator, discriminator)
random_vector = utils.generate_constant_random_vector(
    NOISE_DIM, NUM_EXAMPLES_TO_GENERATE)

utils.train(dataset=train_dataset, epochs=EPOCHS, noise_dim=NOISE_DIM, generator=generator,
            discriminator=discriminator, checkpoint=checkpoint, random_vector=random_vector)

generator.summary()
discriminator.summary()

utils.create_gif()
