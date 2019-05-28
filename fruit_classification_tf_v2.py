from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.20, random_state=42,shuffle=True)


    # Convolutional neural network using 2 filters
    nn = CNN(
            name="test",
            imageWidth=100,
            imageHeight=100,
            hiddenSize=256,
            outputSize=N_class,
            filters=[(5, 5, 3, 20), (3, 3, 20, 50)],
            poolSize=(2,2),
            learningRate=0.0001,
            decay=0.99,
            momentum=0.90
        )

    nn.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=10)
    
    for i in range(len(X_test)):
        j = nn.predictOne(X_test[i])
        p = np.zeros(N_class)
        p[j] = 1
        if j!=np.argmax(y_test[i]):
            print(encoder.inverse_transform([p]),':',encoder.inverse_transform([y_test[i]]))
    
    nn.close()



if __name__ == '__main__':
    main()
