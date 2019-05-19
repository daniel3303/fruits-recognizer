from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

from cnn import CNN
from dataset_util import load_data


def main():
    nn = CNN(
            imageWidth=50,
            imageHeight=50,
            hiddenSize=256,
            outputSize=4,
            filters=[(50, 50, 3, 20), (25, 25, 20, 50), (10, 10, 50, 100)],
            poolSize=(2,2)
        )



if __name__ == '__main__':
    main()
