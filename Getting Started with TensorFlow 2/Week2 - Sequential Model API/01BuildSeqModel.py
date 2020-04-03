import tensorflow as tf 

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Softmax

model = Sequential([
        Flatten(input_shape = (28,28)),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer = 'Adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
)

print(model.loss)
print(model.metrics)
print(model.optimizer)
print(model.weights)