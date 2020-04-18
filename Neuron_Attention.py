# NASNet中NA模块

import tensorflow as tf
from cnn_basenet import CNNBaseModel as cb

def NA(x,is_training=True):
    input_shape = x.get_shape().as_list()
    print('input_shape', input_shape)
    batchsize, dim1, dim2, channels = input_shape
    print(input_shape, '-------------------')
    y = tf.nn.depthwise_conv2d(x, filter=[3, 3, channels, channels], strides=1, padding='SAME', name='dp_conv')
    y = cb.layerbn(y, is_training=is_training, name='bn1')
    y = cb.relu(y,name='relu1')
    y = cb.conv2d(y, channels, 1, name='conv')
    y = tf.nn.sigmoid(y)
    return x * y     #tf.multiply(x, y)