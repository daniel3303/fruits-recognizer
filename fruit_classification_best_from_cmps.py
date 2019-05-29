from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
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
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.50, random_state=0)


    # Convolutional neural network using 2 filters
    nn = CNN(
            name="test",
            imageWidth=100,
            imageHeight=100,
            hiddenSize=256,
            outputSize=N_class,
            filters=[(12, 12, 3, 20), (12, 12, 20, 50)],
            poolSize=(2,2),
            initialization="xavier_glorot",
            regularization="l2"
        )

    cost, accuracy = nn.train(X_train, y_train, X_validation, y_validation, batchSize=128, epochs=9)
    
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.plot(cost) 
    plt.show()
    
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(accuracy)
    plt.show()
    
    count=np.zeros((4,4),dtype=int)
    for i in range(len(X_test)):
        k = np.argmax(y_test[i])
        j = nn.predictOne(X_test[i])
        if j!=k:
            count[k][j] += 1
            
            #To turn on output of individual images misclassified simply change to true
            if False:
                p = np.zeros(N_class)
                p[j] = 1
                print("This picture should be classed as ",encoder.inverse_transform([y_test[i]]))
                plt.imshow(X_test[i])
                plt.show()
                print("It has instead been classified as ",encoder.inverse_transform([p]))
    
    for i in range(N_class):
        p = np.zeros(N_class)
        p[i] = 1
        print(i,':',encoder.inverse_transform([p])) 
    print("Test phase")
    print("mistakes")
    print(count)
    print("Test accuracy:",(len(X_test)-np.sum(count))/len(X_test))
    
    nn.close()



if __name__ == '__main__':
    main()
