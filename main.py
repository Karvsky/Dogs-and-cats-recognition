import os
import cv2
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from matplotlib import pyplot as plt
from model import CNNS_model
from data_import import data_import
from data_normalization import normalization
from split_of_data import splitting_of_data


img_array, img_labels = data_import()

normalized_dataset = normalization()

trainX, testX, trainY, testY = splitting_of_data()

model = CNNS_model()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

model.fit(trainX, trainY, epochs=6, validation_data=(testX, testY))

loss, accuracy = model.evaluate(testX, testY, verbose=1)

print(accuracy)

directory = r'testowa dana\test2.jpg'
image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
new_image = cv2.resize(image, (70, 50))
plt.imshow(new_image, cmap='gray')
plt.show()
image_matrix_float = new_image.astype('float32')
normalized_matrix = image_matrix_float / 255.0
final_image = normalized_matrix.reshape(1, 70, 50, 1)

print(model.predict(final_image))

