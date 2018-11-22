import tensorflow as tf
import utils

EPOCHS = 300
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

tf.enable_eager_execution()


def main():
    # cleaning
    utils.remove_all_files_inside_folder('./results/')
    utils.remove_all_files_inside_folder('./training_checkpoints/')
    # prepare dataset
    (train_images, _), (_, _) = utils.get_fmnist_data()
    train_dataset = utils.normalize_data(train_images)
    # create models
    generator = utils.Generator()
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
