from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

from cnn import CNN
from dataset_util import load_data

# Normalizes images 0 - 255 -> 0 - 1
def normalize(X):
    # X of shape (N, W, H, 3)
    return (X / 255).astype(np.float32)

def main():
    # loads, encodes and normalizes the dataset
    X_train, y_train, X_test, y_test, N_class = load_data()
    encoder = OneHotEncoder(sparse=True)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()
    X_train = normalize(X_train)
    X_train, y_train = shuffle(X_train, y_train)
    X_test = normalize(X_test)



    # Convolutional neural network using 2 filters
    nn = CNN(
            name="test",
            imageWidth=100,
            imageHeight=100,
            hiddenSize=256,
            outputSize=4,
            filters=[(50, 50, 3, 20), (25, 25, 20, 50)],
            poolSize=(2,2),
            learningRate=0.0001,
            decay=0.99,
            momentum=0.90
        )

    nn.train(X_train, y_train, X_test, y_test, batchSize=128, epochs=6)
    nn.close()



if __name__ == '__main__':
    main()
