# https://github.com/pragatiunna/License-Plate-Number-Detection/blob/main/2.%20License%20Plate%20Detection%20(using%20YOLOv3).ipynb# dependencies
# NumPy versions below 1.17 may be incompatible with some other 
# packages, so you may need to replace your current version with 
# an earlier one in order to run this notebook as-is. 
# !pip uninstall numpy --yes
# !pip install "numpy<1.17"
#pip install ipython
#pip install matplotlib
#pip install opencv-python
#pip install argparse


from IPython.display import Image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec 

import cv2
import argparse
import sys
import numpy as np
import pandas as pd
import os.path

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential

#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import tensorflow as tf

class StopAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        super(StopAtAccuracy, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        # Puedes cambiar 'accuracy' por 'val_accuracy' si prefieres monitorear la validación
        #current_acc = logs.get('accuracy')
        current_acc = logs.get('val_accuracy')
        
        if current_acc is not None and current_acc >= self.target_accuracy:
            print(f"\n[INFO] Se alcanzó el {self.target_accuracy * 100}% de accuracy en la época {epoch + 1}. Deteniendo el entrenamiento.")
            self.model.stop_training = True

# Ejemplo de uso: detenerse al llegar al 95% de accuracy
callback_precision = StopAtAccuracy(target_accuracy=0.99)

# https://github.com/pragatiunna/License-Plate-Number-Detection/blob/main/1.%20License%20Plate%20Detection%20(using%20Contours).ipynb

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data'
train_generator = train_datagen.flow_from_directory(
        path+'/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=80,
        class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(
        path+'/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28 batch_size=80,
        class_mode='sparse')

#K.clear_session()
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

#model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics='accuracy') #MOD
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

model.summary()

#batch_size = 1 #MOD
batch_size = 80
result = model.fit(
      train_generator,
      steps_per_epoch = train_generator.samples // batch_size,
      validation_data = validation_generator, 
      epochs = 400, verbose=1,
      callbacks=None
      #callbacks=[callback_precision]
      )

fig = plt.figure(figsize=(14,5))
grid=gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
fig.add_subplot(grid[0])
plt.plot(result.history['accuracy'], label='training accuracy')
plt.plot(result.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#fig.add_subplot(grid[1])
plt.plot(result.history['loss'], label='training loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Save the weights
model.save_weights('OCR.weights.h5')




