import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

data = tf.cast(np.random.random((1000, 32)), tf.float32)
labels = tf.cast(np.random.random((1000, 10)), tf.float32)

print("---------- Building a Model ----------")
inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor
# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("---------- Training ----------")
# Trains for 5 epochs
model.fit(data, labels, epochs=5, steps_per_epoch=30)