import numpy as np 
import tensorflow as tf

#Training a feedforward neural network for image classification
print('Loading data....\n')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('MNIST dataset loaded.\n')

x_train = x_train/255.

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')

])

model.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

print('Training model...\n')
model.fit(x_train, y_train, epochs = 3, batch_size=32)

print('Model trained successfully')
