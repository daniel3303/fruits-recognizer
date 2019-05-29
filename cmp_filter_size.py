from __future__ import print_function, division

import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.50, random_state=0)

    """ IN THIS SECTION WE COMPARE DIFFENT FILTER SIZES """
    # 3x3 filters
    nn3 = CNN(
        name="small_filter_size",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=N_class,
        filters=[(3, 3, 3, 20), (3, 3, 20, 50)],
        poolSize=(2, 2),
        initialization="xavier_glorot",
        regularization="dropout"
    )
    cost3, accuracy3 = nn3.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=10)

    # 6X6 filters
    nn6 = CNN(
        name="medium_filter_size",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=N_class,
        filters=[(6, 6, 3, 20), (6, 6, 20, 50)],
        poolSize=(2, 2),
        initialization="xavier_glorot",
        regularization="dropout"
    )

    cost6, accuracy6 = nn6.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=10)

    # 12x12 filters
    nn12 = CNN(
        name="large_filter_size",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=N_class,
        filters=[(12, 12, 3, 20), (12, 12, 20, 50)],
        poolSize=(2, 2),
        initialization="xavier_glorot",
        regularization="dropout"
    )

    cost12, accuracy12 = nn12.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=10)
    
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.plot(cost3, label='3') 
    plt.plot(cost6, label='6')
    plt.plot(cost12, label='12')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(accuracy3, label='3') 
    plt.plot(accuracy6, label='6')
    plt.plot(accuracy12, label='12')
    plt.legend(loc='upper left')
    plt.show()
    
    
    count3=np.zeros((4,4),dtype=int)
    count6=np.zeros((4,4),dtype=int)
    count12=np.zeros((4,4),dtype=int)
    for i in range(len(X_test)):
        k = np.argmax(y_test[i])
        j3 = nn3.predictOne(X_test[i])
        j6 = nn6.predictOne(X_test[i])
        j12 = nn12.predictOne(X_test[i])
        if j3!=k:
            count3[k][j3] += 1
        if j6!=k:
            count6[k][j6] += 1
        if j12!=k:
            count12[k][j12] += 1
    
    for i in range(N_class):
        p = np.zeros(N_class)
        p[i] = 1
        print(i,':',encoder.inverse_transform([p])) 
    print("Test phase")
    print("------")
    print("3:")
    print("mistakes")
    print(count3)
    print("Test accuracy:",(len(X_test)-np.sum(count3))/len(X_test))
    print("------")
    print("6:")
    print("mistakes")
    print(count6)
    print("Test accuracy:",(len(X_test)-np.sum(count6))/len(X_test))
    print("------")
    print("12:")
    print("mistakes")
    print(count12)
    print("Test accuracy:",(len(X_test)-np.sum(count12))/len(X_test))
    
    
    


if __name__ == '__main__':
    main()