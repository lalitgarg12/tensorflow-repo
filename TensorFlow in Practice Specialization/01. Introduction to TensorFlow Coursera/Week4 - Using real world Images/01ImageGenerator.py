import tensorflow as tf

#One feature of the image generator is that you can point it at a directory 
#and then the sub-directories of that will automatically generate labels for you.

#When you put sub-directories in these for horses and humans and store the requisite images in there, the 
#image generator can create a feeder for those images and auto label them for you.

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'Mydir'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)

#The nice thing about this code is that the images are resized for you as they're loaded.
#So you don't need to preprocess thousands of images on your file system. 
#But you could have done that if you wanted to. The advantage of doing it at runtime like this is that 
#you can then experiment with different sizes without impacting your source data. 

#Defining a Convnet to use complex images
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(16,(3.3), activation = 'relu', input_shape=(300,300,3)),
    MaxPooling2D(2,2),
    Conv2D(32,(3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

from keras.optimizers import RMSprop

model.compile(loss = 'binary_crossentropy',
            optimizer=RMSprop(learning_rate=0.001),
            metrics=['accuracy'])

validation_generator = 'valid_gen'

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8,
    verbose=2
)

#Using model.fit_generator instead of model.fit because we are using generator instead of datasets.
#First parameter is training generator that we set up earlier. 
#This streams the images from the training directory.
#steps_per_epoch - Remember the batch size we created. There are 1024 images in training directory,
#So we are loading them in 128 at a time. So in order to load them, we need to do 8 batches.
#Validation_data also comes from validation generator which we create earlier.
#It had 256 images and we handle them in batches of 32, so we set this at 8

import numpy as np 
#from google.colab import files
from keras.preprocessing import image

#uploaded = files.upload()

# for fn in uploaded.keys():
#     #predicting images
#     path = '/content/' + fn
#     img = image.load_img(path, target_size=(300,300))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(classes[0])
#     if classes[0]>0.5:
#         print(fn + "is a human")
#     else:
#         print(fn + "is a horse")

#So these parts are specific to Colab, they are what gives you the button 
#that you can press to pick one or more images to upload. 
#The image paths then get loaded into this list called uploaded. 
#The loop then iterates through all of the images in that collection. 
#And you can load an image and prepare it to input into the model with this code. 
#Take note to ensure that the dimensions match the input dimensions that you specified when designing the model.
