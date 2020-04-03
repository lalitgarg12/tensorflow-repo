import numpy as np 
import tensorflow as tf 
from tensorflow import keras

#Loading dataset from keras API by load_data method
fashion_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Defining the model by using keras sequential API
model = keras.Sequential([
                         keras.layers.Flatten(input_shape = (28,28)),
                         keras.layers.Dense(128, activation = tf.nn.relu),
                         keras.layers.Dense(10, activation = tf.nn.softmax)
])

#Showing the single train data image
import matplotlib.pyplot as plt
plt.imshow(training_images[42])
print(training_labels[42])
print(training_images[42])

#Normalizing data input
training_images = training_images / 255.0
test_images = test_images / 255.0

#Compiling the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

#Fitting the model (doing training on training dataset)
model.fit(training_images, training_labels, epochs = 5)

#Evaluating on test set
model.evaluate(test_images, test_labels)