from __future__ import print_function, division

import numpy as np

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

    """ IN THIS SECTION WE COMPARE DIFFENT FILTER SIZES """
    # 3x3 filters
    nn = CNN(
        name="small_filter_size",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=4,
        filters=[(3, 3, 3, 20), (3, 3, 20, 50)],
        poolSize=(2, 2),
        learningRate=0.0001,
        decay=0.99,
        momentum=0.90
    )
    cost, accuracy = nn.train(X_train, y_train, X_test, y_test, batchSize=128, epochs=5)
    nn.close()

    # 6X6 filters
    nn = CNN(
        name="medium_filter_size",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=4,
        filters=[(6, 6, 3, 20), (6, 6, 20, 50)],
        poolSize=(2, 2),
        learningRate=0.0001,
        decay=0.99,
        momentum=0.90
    )

    cost, accuracy = nn.train(X_train, y_train, X_test, y_test, batchSize=128, epochs=5)
    nn.close()

    # 12x12 filters
    nn = CNN(
        name="large_filter_size",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=4,
        filters=[(12, 12, 3, 20), (12, 12, 20, 50)],
        poolSize=(2, 2),
        learningRate=0.0001,
        decay=0.99,
        momentum=0.90
    )

    cost, accuracy = nn.train(X_train, y_train, X_test, y_test, batchSize=128, epochs=5)
    nn.close()

    print(cost)
    print(accuracy)


if __name__ == '__main__':
    main()