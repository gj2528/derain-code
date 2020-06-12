'''
this block are not offcial implement of non_local_block in paper
Non-local Neural Networks https://arxiv.org/pdf/1711.07971.pdf

'''

import tensorflow as tf
from cnn_basenet import CNNBaseModel as cb



def non_local_block(input_tensor, computation_compression=2,is_training=True):

    input_shape = input_tensor.get_shape().as_list()
    print('input_shape',input_shape)
    batchsize, dim1, dim2, channels = input_shape
    print(input_shape,'-------------------')

    theta = cb.conv2d(input_tensor, channels, 1, name='theta')
    theta = cb.layerbn(theta, is_training=is_training, name='bn1')   #回学校可以去掉试试
    theta = cb.relu(theta,name='relu1')

    phi = cb.conv2d(input_tensor, channels, 1, name='phi')
    phi = cb.layerbn(phi, is_training=is_training, name='bn2')     #回学校可以去掉试试
    phi = cb.relu(phi, name='relu2')
    # phi = tf.reshape(phi,shape=[batchsize, dim2, dim1, channels])

    theta_x = tf.reshape(theta, [batchsize, channels, -1])
    theta_x = tf.transpose(theta_x, [0,2,1])
    phi_x = tf.reshape(phi, [batchsize, channels, -1])

    f = tf.matmul(theta_x, phi_x)
    f = tf.nn.softmax(f)

    g = cb.conv2d(input_tensor, channels, 1, name='g')
    g = cb.layerbn(g, is_training=is_training, name='bn3')    #回学校可以去掉试试
    g = cb.relu(g, name='relu3')

    g_x = tf.reshape(g, [batchsize,channels, -1])
    g_x = tf.transpose(g_x, [0,2,1])

    y = tf.matmul(f, g_x)

    print('y=', y)


    y = cb.conv2d(y, channels, kernel_size=3, name='y')
    print('y=', y)

    residual = input_tensor + y

    return residual
