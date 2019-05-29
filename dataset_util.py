import numpy as np

import cv2
import os


def list_files(baseDir):
    return [dir[0]+"/"+file for dir in os.walk(baseDir) for file in dir[2] if os.path.isfile(dir[0]+"/"+file) and file.endswith(".jpg")]

def load_image(path):
    bgr_img = cv2.imread(path)
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    return cv2.merge([r,g,b]) 


# Returns X_train, y_train, X_test, y_test
def load_data():
    # Train data
    X_train = []
    y_train = []
    classes = [name for name in os.listdir(path='data/train') if os.path.isdir("data/train/"+name)]

    print("Loading train data...")
    for cn in classes:
        files = list_files("data/train/"+cn)
        for file in files:
            X_train.append(load_image(file))
            y_train.append(cn)
    print(str(len(y_train))+" images found.")

    # Test data
    X_test = []
    y_test = []
    classes = [name for name in os.listdir(path='data/test') if os.path.isdir("data/test/" + name)]

    print("Loading test data...")
    for cn in classes:
        files = list_files("data/test/" + cn)
        for file in files:
            X_test.append(load_image(file))
            y_test.append(cn)
    print(str(len(y_test)) + " images found.")

    print("Classes found: ")
    for c in sorted(classes):
        print("   - "+c)

    return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(classes))