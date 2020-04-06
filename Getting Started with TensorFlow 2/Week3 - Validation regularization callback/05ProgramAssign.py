from sklearn import datasets, model_selection
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras import regularizers
import tensorflow as tf 
from keras.callbacks import EarlyStopping

iris_data = datasets.load_iris()
print(iris_data.keys())

def read_in_and_split_data(iris_data):
    """
    This function takes the Iris dataset as loaded by sklearn.datasets.load_iris(), and then 
    splits so that the training set includes 90% of the full dataset, with the test set 
    making up the remaining 10%.
    Your function should return a tuple (train_data, test_data, train_targets, test_targets) 
    of appropriately split training and test data and targets.
    
    If you would like to import any further packages to aid you in this task, please do so in the 
    Package Imports cell above.
    """
    iris_data = datasets.load_iris()
    data = iris_data['data']
    targets = iris_data['target']
    train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1) 
    return (train_data, test_data, train_targets, test_targets)

iris_data = datasets.load_iris()
train_data, test_data, train_targets, test_targets = read_in_and_split_data(iris_data)
print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)
print(test_targets.shape)

# Convert targets to a one-hot encoding

train_targets = tf.keras.utils.to_categorical(np.array(train_targets))
test_targets = tf.keras.utils.to_categorical(np.array(test_targets))

def get_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_initializer='he_uniform',bias_initializer='ones',input_shape=(input_shape)),
        Dense(128,activation='relu'),
        Dense(128,activation='relu'),
        Dense(128,activation='relu'),
        Dense(128,activation='relu'),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(64,activation='relu'),
        Dense(3, activation='softmax')
    ])

    return model 

def compile_model(model):
    """
    This function takes in the model returned from your get_model function, and compiles it with an optimiser,
    loss function and metric.
    Compile the model using the Adam optimiser (with learning rate set to 0.0001), 
    the categorical crossentropy loss function and accuracy as the only metric. 
    Your function doesn't need to return anything; the model will be compiled in-place.
    """
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

def train_model(model, train_data, train_targets, epochs):
    """
    This function should train the model for the given number of epochs on the 
    train_data and train_targets. 
    Your function should return the training history, as returned by model.fit.
    """
    history = model.fit(train_data, train_targets, epochs=epochs, 
                        batch_size=40, validation_split=0.15,verbose=False)
    
    return history


def get_regularised_model(input_shape, dropout_rate, weight_decay):
    """
    This function should build a regularised Sequential model according to the above specification. 
    The dropout_rate argument in the function should be used to set the Dropout rate for all Dropout layers.
    L2 kernel regularisation (weight decay) should be added using the weight_decay argument to 
    set the weight decay coefficient in all Dense layers that use L2 regularisation.
    Ensure the weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument input_shape.
    Your function should return the model.
    """
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer = regularizers.l2(weight_decay), kernel_initializer='he_uniform',bias_initializer='ones',input_shape=(input_shape)),
        Dense(128,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(128,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(128,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(128,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        BatchNormalization(),
        Dense(64,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(64,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(64,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(64,activation='relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dense(3, activation='softmax')
    ])

    return model 

def get_callbacks():
    """
    This function should create and return a tuple (early_stopping, learning_rate_reduction) callbacks.
    The callbacks should be instantiated according to the above requirements.
    """
    early_stopping = EarlyStopping(monitor = 'val_loss',
                                  patience = 30,
                                  min_delta=0.01,
                                  mode='min')
    
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                  factor=0.2,
                                                                  patience=20)
    
    return (early_stopping, learning_rate_reduction)

