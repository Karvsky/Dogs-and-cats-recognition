import numpy as np
from sklearn.model_selection import train_test_split
from data_import import data_import
from data_normalization import normalization

def splitting_of_data():

    img_array, img_labels = data_import()

    normalized_dataset = normalization()

    X = np.array(normalized_dataset)
    Y = np.array(img_labels)

    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=12)

    return trainX, testX, trainY, testY