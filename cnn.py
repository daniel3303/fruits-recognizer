import tensorflow as tf
import numpy as np


class CNN:
    """
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
    """
    def __init__(self, imageWidth, imageHeight, hiddenSize, outputSize, filters=None, poolSize=(2, 2)):
        self.annHiddenSize = hiddenSize
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.annOutputSize = outputSize
        self.poolSize = [1, poolSize[0], poolSize[1], 1]
        self.filters = filters


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
        # vanilla ANN weights, the weights are initialized according to the Xavier/Glorot's initialization method
        # original paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        Wh_init = np.random.randn(self.annInputSize, self.annHiddenSize) / np.sqrt(self.annInputSize + self.annHiddenSize)
        bh_init = np.zeros(self.annHiddenSize, dtype=np.float32)
        Wo_init = np.random.randn(self.annHiddenSize, self.annOutputSize) / np.sqrt(self.annHiddenSize + self.annOutputSize)
        bo_init = np.zeros(self.annOutputSize, dtype=np.float32)

        Wh = tf.Variable(Wh_init.astype(np.float32))
        bh = tf.Variable(bh_init.astype(np.float32))
        Wo = tf.Variable(Wo_init.astype(np.float32))
        bo = tf.Variable(bo_init.astype(np.float32))

        Z_last = self.convLayers[-1].Z
        Z_last_shape = Z_last.get_shape().as_list()

        Zr = tf.reshape(Z_last, [-1, np.prod(Z_last_shape[1:])])
        Zh = tf.nn.relu(tf.matmul(Zr, Wh) + bh)
        self.Yish = tf.matmul(Zh, Wo) + bo


        # cost function
        cost = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.Yish,
                labels=self.T
            )
        )

        # train function
        self.trainOp = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)


class ConvolutionalLayer:
    def __init__(self, X, filter, poolSize):
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
        w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
        return w.astype(np.float32)

    def createConvLayer(self):
        filter = self.filter
        W_init = self.initFilter(filter)
        b_init = np.zeros(filter[-1], dtype=np.float32)  # one bias per output feature map

        self.W = tf.Variable(W_init.astype(np.float32))
        self.b = tf.Variable(b_init.astype(np.float32))

        self.Z = self.convPool(self.X, self.W, self.b)


