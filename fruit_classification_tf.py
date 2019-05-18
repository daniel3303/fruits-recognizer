from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

from dataset_util import load_data


def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape):
    # w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2]) / np.prod(poolsz))
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)


# Normalizes images 0 - 255 -> 0 - 1
def normalize(X):
    # X of shape (N, W, H, 3)
    return (X / 255).astype(np.float32)



def main():
    X_train, y_train, X_test, y_test, N_class = load_data()
    img_width = X_train.shape[1]
    img_height = X_train.shape[1]
    img_size = img_width * img_height

    # Encode targets
    encoder = OneHotEncoder(sparse=True)

    y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()

    # Need to scale! don't leave as 0..255
    # Also need indicator matrix for cost calculation
    X_train = normalize(X_train)
    X_train, y_train = shuffle(X_train, y_train)

    X_test = normalize(X_test)

    # gradient descent params
    max_iter = 6
    print_period = 10
    N = X_train.shape[0]
    batch_sz = 128
    n_batches = N // batch_sz

    # initial weights
    M = 256
    K = N_class
    poolsz = (2, 2)

    W1_shape = (5, 5, 3, 20)  # (filter_width, filter_height, num_color_channels, num_feature_maps)
    W1_init = init_filter(W1_shape)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)  # one bias per output feature map

    W2_shape = (5, 5, 20, 50)  # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
    W2_init = init_filter(W2_shape)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # vanilla ANN weights
    W3_init = np.random.randn(W2_shape[-1] * 25 * 25, M) / np.sqrt(W2_shape[-1] * 25 * 25 + M)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)

    # define variables and expressions
    # using None as the first shape element takes up too much RAM unfortunately
    X = tf.placeholder(tf.float32, shape=(None, img_width, img_height, 3), name='X')
    T = tf.placeholder(tf.int32, shape=(None, N_class), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))

    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, [-1, np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)
    Yish = tf.matmul(Z3, W4) + b4

    cost = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=Yish,
            labels=T
        )
    )

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    t0 = datetime.now()
    LL = []
    W1_val = None
    W2_val = None
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        error = []
        accuracy = []
        for i in range(max_iter):
            epochErr = 0

            for j in range(n_batches):
                print(
                    "\rEpoch " + str(i + 1) + " of " + str(max_iter) + " | Training batch " + str(j + 1) + " of " + str(
                        n_batches), end="")

                Xbatch = X_train[j * batch_sz:(j * batch_sz + batch_sz), ]
                Ybatch = y_train[j * batch_sz:(j * batch_sz + batch_sz), ]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                epochErr += session.run(cost, feed_dict={X: Xbatch, T: Ybatch})

            # after one epoch calculates the accuracy
            testSucc = 0
            testAcc = 0
            for k in range(0,len(y_test) // batch_sz):
                Xbatch = X_test[k * batch_sz:(k + 1) * batch_sz,]
                Ybatch = y_test[k * batch_sz:(k + 1) * batch_sz,]

                prediction = session.run(predict_op, feed_dict={X: Xbatch, T: Ybatch})
                testSucc += np.sum(prediction == np.argmax(Ybatch))
                testAcc = testSucc/len(y_test)

            error.append(epochErr)
            accuracy.append(testAcc)
            print(" | Epoch error: {0:.0f}".format(epochErr)+" | Accuracy: {0:.2f}".format(testAcc))

        W1_val = W1.eval()
        W2_val = W2.eval()
    print("Elapsed time:", (datetime.now() - t0))
    # plt.plot(LL)
    # plt.show()
    print(error)


if __name__ == '__main__':
    main()
