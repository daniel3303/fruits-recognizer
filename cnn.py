import tensorflow as tf
import numpy as np

class CNN:
    """
        name:
            - an unique name used to create a variable scope
        imageWidth:
            - the width of the images we are going to train
        imageHeight:
            - the height of the images we are going to train
        hiddenSize:
            - number of neurons in the ann hidden layer
        outputSize:
            - number of different classes in the dataset
        filters:
            - a list of convolution filters
            - example: [(filter_width, filter_height, num_color_channels, num_feature_maps)]
        poolSize:
            - max pooling size
            - example: (2,2) performs max pooling over a 2x2 matrix
        learningRate:
            - the learning rate
        decay:
            - RMSProp decay. How much to decay the current momentum
        momentum:
            - the momentum
        initialization:
            - How the neural networks weights are initialized. Note that the bias is always initialized to zero.
            - Available options:
                - xavier_glorot     | (default) original paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
                - zeros             | matrix of zeros
                - random_normal     | sample from a Guassian with mean 0 and deviation 1
        regularization:
            - Available options:
                - None              | (default)
                - dropout
                - l1
                - l2
    """
    def __init__(self, name, imageWidth, imageHeight, hiddenSize, outputSize, filters=None, poolSize=(2, 2), learningRate = 0.0001, decay=0.99, momentum=0.9, initialization = "xavier_glorot", regularization = None):
        self.regularization = regularization
        self.initialization = initialization
        self.name = name
        self.momentum = momentum
        self.decay = decay
        self.learningRate = learningRate
        self.annHiddenSize = hiddenSize
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.annOutputSize = outputSize
        self.poolSize = [1, poolSize[0], poolSize[1], 1]
        self.filters = filters
        self.session = tf.Session()

        with tf.name_scope(self.name):
            # input and target placeholders
            self.X = tf.placeholder(tf.float32, shape=(None, imageWidth, imageHeight, 3), name='X')
            self.T = tf.placeholder(tf.int32, shape=(None, self.annOutputSize), name='T')

            # layers
            self.convLayers = []

            for filter in self.filters:
                if len(self.convLayers) == 0:
                    layer = ConvolutionalLayer(self.X, filter, self.poolSize)
                else:
                    layer = ConvolutionalLayer(self.convLayers[-1].Z, filter, self.poolSize)
                self.convLayers.append(layer)

            # calculate the ann number of input neurons based on the previous pooling layers
            Z_last = self.convLayers[-1].Z
            Z_last_shape = Z_last.get_shape().as_list()
            self.annInputSize = np.prod(Z_last_shape[1:])

            # ann layers (we use just one hidden layer for simplicity, we can tweak the number of neurons in it)
            # in total we have 3 ann layers, the input layer, the hidden layer and the output layer
            Wh_init = self.init_weight(self.annInputSize, self.annHiddenSize)
            bh_init = np.zeros(self.annHiddenSize, dtype=np.float32)
            Wo_init = self.init_weight(self.annHiddenSize, self.annOutputSize)
            bo_init = np.zeros(self.annOutputSize, dtype=np.float32)

            Wh = tf.Variable(Wh_init.astype(np.float32), name="hidden_weights")
            bh = tf.Variable(bh_init.astype(np.float32), name="hidden_bias")
            Wo = tf.Variable(Wo_init.astype(np.float32), name="output_weights")
            bo = tf.Variable(bo_init.astype(np.float32), name="output_bias")

            Z_last = self.convLayers[-1].Z
            Z_last_shape = Z_last.get_shape().as_list()

            Zr = tf.reshape(Z_last, [-1, np.prod(Z_last_shape[1:])])
            Zh = tf.nn.relu(tf.matmul(Zr, Wh) + bh)

            if self.regularization == "dropout":
                Zh = tf.layers.dropout(inputs=Zh, rate=0.4)

            self.Yish = tf.matmul(Zh, Wo) + bo


            # cost function
            loss = 0.0
            if self.regularization == "l2":
                vars = tf.trainable_variables(scope=self.name)
                loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 0.001
            elif self.regularization == "l1":
                l1reg = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
                vars = tf.trainable_variables(scope=self.name)  # all vars of your graph
                loss = tf.contrib.layers.apply_regularization(l1reg, vars)


            self.cost = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.Yish,
                    labels=self.T
                )
            ) + loss

            # train operation
            self.trainOp = tf.train.RMSPropOptimizer(learning_rate=self.learningRate, decay=self.decay, momentum=self.momentum).minimize(self.cost)

            # predict operation
            self.predictOp = tf.argmax(self.Yish, 1)

        # initialize variables
        init = tf.global_variables_initializer()

        self.session.run(init)

    def init_weight(self, *shape):
        if self.initialization == "zeros":
            w = np.zeros(shape)
        elif self.initialization == "random_normal":
            w = np.random.randn(*shape)
        else: # default xavier_glorot
            w = np.random.randn(*shape) / np.sqrt(np.sum(shape))
        return w.astype(np.float32)

    def train(self, X_train, y_train, X_test, y_test, batchSize = 128, epochs = 5):
        N = len(X_train)
        n_batches = N//batchSize

        error = []
        accuracy = []
        for e in range(0, epochs):
            epochErr = 0
            for b in range(0, n_batches):
                print(
                    "\rEpoch " + str(e + 1) + " of " + str(epochs) + " | Training batch " + str(b + 1) + " of " + str(
                        n_batches), end="")

                Xbatch = X_train[b * batchSize:(b * batchSize + batchSize), ]
                Ybatch = y_train[b * batchSize:(b * batchSize + batchSize), ]

                self.session.run(self.trainOp, feed_dict={self.X: Xbatch, self.T: Ybatch})
                epochErr += self.session.run(self.cost, feed_dict={self.X: Xbatch, self.T: Ybatch})

            # after one epoch calculates the accuracy
            testSucc = 0
            testAcc = 0
            for k in range(0, len(y_test) // batchSize):
                Xbatch = X_test[k * batchSize:(k + 1) * batchSize, ]
                Ybatch = y_test[k * batchSize:(k + 1) * batchSize, ]

                prediction = self.session.run(self.predictOp, feed_dict={self.X: Xbatch})
                testSucc += np.sum(prediction == np.argmax(Ybatch))
                testAcc = testSucc / len(y_test)

            error.append(epochErr)
            accuracy.append(testAcc)
            print(" | Epoch error: {0:.0f}".format(epochErr) + " | Accuracy: {0:.2f}".format(testAcc))

        return (error, accuracy)

    def predict(self, X):
        return self.session.run(self.predictOp, feed_dict={self.X: X})

    def predictOne(self, X):
        return self.session.run(self.predictOp, feed_dict={self.X: np.array([X])})[0]


    def close(self):
        self.session.close()


class ConvolutionalLayer:
    def __init__(self, X, filter, poolSize, initialization = None):
        self.initialization = initialization
        self.X = X
        self.Z = None
        self.poolSize = poolSize
        self.filter = filter

        # tensorflow variables
        self.W = None
        self.b = None

        # creates self.Y
        self.createConvLayer()


    def convPool(self, X, W, b):
        # just assume pool size is (2,2) because we need to augment it with 1s
        conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, b)
        pool_out = tf.nn.max_pool(conv_out, ksize=self.poolSize, strides=self.poolSize, padding='SAME')
        return tf.nn.relu(pool_out)

    def initFilter(self, shape):
        if self.initialization == "zeros":
            w = np.zeros(shape)
        elif self.initialization == "random_normal":
            w = np.random.randn(*shape)
        else: # default xavier_glorot
            w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
        return w.astype(np.float32)

    def createConvLayer(self):
        filter = self.filter
        W_init = self.initFilter(filter)
        b_init = np.zeros(filter[-1], dtype=np.float32)  # one bias per output feature map

        self.W = tf.Variable(W_init.astype(np.float32))
        self.b = tf.Variable(b_init.astype(np.float32))

        self.Z = self.convPool(self.X, self.W, self.b)

