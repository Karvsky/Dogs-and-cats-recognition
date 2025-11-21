import os
import cv2
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from matplotlib import pyplot as plt
from model import CNNS_model
from data_import import data_import
from data_normalization import normalization
from split_of_data import splitting_of_data
import time

class DL_MODEL:
    def __init__(self):

        img_array, img_labels = data_import()

        normalized_dataset = normalization()

        trainX, testX, trainY, testY = splitting_of_data()

        self.model = CNNS_model()

        self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

        self.model.fit(trainX, trainY, epochs=6, validation_data=(testX, testY))

        loss, accuracy = self.model.evaluate(testX, testY, verbose=1)

        pass

    def image_decision(self, route):

        image = cv2.imread(route, cv2.IMREAD_GRAYSCALE)
        new_image = cv2.resize(image, (120, 120))
        image_matrix_float = new_image.astype('float32')
        normalized_matrix = image_matrix_float / 255.0
        final_image = normalized_matrix.reshape(1, 120, 120, 1)

        decision = self.model.predict(final_image)

        if decision[0][0] < 0.5:
            return "Zdjęcie przedstawia kota"
        elif decision[0][0] > 0.5:
            return "Zdjęcie przedstawia psa"
        else:
            return "Nie wiadomo co zdjęcie przedstawia"


