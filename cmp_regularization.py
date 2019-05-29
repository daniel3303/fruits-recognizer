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

    """ IN THIS SECTION WE COMPARE DIFFENT REGULARIZATIONS """

    nn1 = CNN(
        name="No_regularization",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=N_class,
        filters=[(3, 3, 3, 20), (3, 3, 20, 50)],
        poolSize=(2, 2),
        initialization="xavier_glorot",
        regularization="None"
    )
    cost1, accuracy1 = nn1.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=20)


    nn2 = CNN(
        name="dropout_regularization",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=N_class,
        filters=[(3, 3, 3, 20), (3, 3, 20, 50)],
        poolSize=(2, 2),
        initialization="xavier_glorot",
        regularization="dropout"
    )

    cost2, accuracy2 = nn2.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=20)


    nn3 = CNN(
        name="l1_regularization",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=N_class,
        filters=[(3, 3, 3, 20), (3, 3, 20, 50)],
        poolSize=(2, 2),
        initialization="xavier_glorot",
        regularization="l1"
    )

    cost3, accuracy3 = nn3.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=20)
    
    
    nn4 = CNN(
        name="l2_regularization",
        imageWidth=100,
        imageHeight=100,
        hiddenSize=256,
        outputSize=N_class,
        filters=[(3, 3, 3, 20), (3, 3, 20, 50)],
        poolSize=(2, 2),
        initialization="xavier_glorot",
        regularization="l2"
    )

    cost4, accuracy4 = nn4.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=20)
    
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.plot(cost1, label='None')
    plt.plot(cost2, label='dropout')
    plt.plot(cost3, label='l1')
    plt.plot(cost4, label='l2')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(accuracy1, label='None')
    plt.plot(accuracy2, label='dropout')
    plt.plot(accuracy3, label='l1')
    plt.plot(accuracy4, label='l2')
    plt.legend(loc='upper left')
    plt.show()
    
    
    count1=np.zeros((4,4),dtype=int)
    count2=np.zeros((4,4),dtype=int)
    count3=np.zeros((4,4),dtype=int)
    count4=np.zeros((4,4),dtype=int)
    for i in range(len(X_test)):
        k = np.argmax(y_test[i])
        j1 = nn1.predictOne(X_test[i])
        j2 = nn2.predictOne(X_test[i])
        j3 = nn3.predictOne(X_test[i])
        j4 = nn4.predictOne(X_test[i])
        if j1!=k:
            count1[k][j1] += 1
        if j2!=k:
            count2[k][j2] += 1
        if j3!=k:
            count3[k][j3] += 1
        if j4!=k:
            count4[k][j4] += 1
    
    for i in range(N_class):
        p = np.zeros(N_class)
        p[i] = 1
        print(i,':',encoder.inverse_transform([p])) 
    print("Test phase")
    print("------")
    print("None:")
    print("mistakes")
    print(count1)
    print("Test accuracy:",(len(X_test)-np.sum(count1))/len(X_test))
    print("------")
    print("dropout:")
    print("mistakes")
    print(count2)
    print("Test accuracy:",(len(X_test)-np.sum(count2))/len(X_test))
    print("------")
    print("l1:")
    print("mistakes")
    print(count3)
    print("Test accuracy:",(len(X_test)-np.sum(count3))/len(X_test))
    print("------")
    print("l2:")
    print("mistakes")
    print(count4)
    print("Test accuracy:",(len(X_test)-np.sum(count4))/len(X_test))
    
    


if __name__ == '__main__':
    main()