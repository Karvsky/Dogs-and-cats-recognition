import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def CNNS_model():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(70, 50, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())

    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))

    return model
