import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

#Equation of a line y = 2x - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs = 500)

print(model.predict([10.0]))

# There are two function roles that you should be aware of though and these are loss functions and optimizers. 
# This code defines them. I like to think about it this way. The neural network has no idea of the relationship 
# between X and Y, so it makes a guess. Say it guesses Y equals 10X minus 10. It will then use the data that it knows about, 
# that's the set of Xs and Ys that we've already seen to measure how good or how bad its guess was. 
# The loss function measures this and then gives the data to the optimizer which figures out the next guess. 
# So the optimizer thinks about how good or how badly the guess was done using the data from the loss function. 
# Then the logic is that each guess should be better than the one before. As the guesses get better and better, 
# an accuracy approaches 100 percent, the term convergence is used.