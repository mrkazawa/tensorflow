import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

print("---------- Output Version ----------")
print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = tf.cast(np.random.random((1000, 32)), tf.float32)
labels = tf.cast(np.random.random((1000, 10)), tf.float32)

val_data = tf.cast(np.random.random((100, 32)), tf.float32)
val_labels = tf.cast(np.random.random((100, 10)), tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

print("---------- Training ----------")
model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

print("---------- Evaluataing and Predicting ----------")
model.evaluate(data, labels, steps=1000)
model.evaluate(dataset, steps=30)
result = model.predict(data, steps=30)
print(result.shape)