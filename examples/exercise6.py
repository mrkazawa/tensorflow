import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import json
import pprint

print("----------- Saving Data -----------")

model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')

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

json_string = model.to_json()
pprint.pprint(json.loads(json_string))