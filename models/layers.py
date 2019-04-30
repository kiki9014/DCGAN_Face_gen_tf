import tensorflow as tf
import numpy as np

def conv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding, use_bias=use_bias, kernel_initializer=tf.initializers.random_normal(0.0, 0.02))

def deconv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding, use_bias=use_bias, kernel_initializer=tf.initializers.random_normal(0.0, 0.02))

def batchNormalization(x, is_train):
    """
    Add a new batchNormalization layer.
    :param x: tf.Tensor, shape: (N, H, W, C) or (N, D)
    :param is_train: tf.placeholder(bool), if True, train mode, else, test mode
    :return: tf.Tensor.
    """
    return tf.layers.batch_normalization(x, training=is_train, momentum=0.9, epsilon=1e-5, center=True, scale=True)


def conv_bn_lrelu(x, filters, kernel_size, is_train, strides=(1, 1), padding='SAME', bn=True, alpha=0.2):
    """
    Add conv + bn + Leaky Relu layers.
    see conv_layer and batchNormalization function
    If you want relu, just change alpha to 0
    If you don't want activation layer, change alpha to 1.0
    """
    conv = conv_layer(x, filters, kernel_size, strides, padding, use_bias=True)
    if bn:
        _bn = batchNormalization(conv, is_train)
    else:
        _bn = conv
    return tf.nn.leaky_relu(_bn, alpha)
    
def deconv_bn_relu(x, filters, kernel_size, is_train, strides=(1, 1), padding='SAME', bn=True, relu=True):
    """
    Add conv + bn + Relu layers.
    see conv_layer and batchNormalization function
    """
    deconv = deconv_layer(x, filters, kernel_size, strides, padding, use_bias=True)
    if bn:
        _bn = batchNormalization(deconv, is_train)
    else:
        _bn = deconv
    if relu:
        return tf.nn.relu(_bn)
    else:
        return _bn


def fc_layer(x, out_dim, **kwargs):
    """
    Add a new fully-connected layer.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, the dimension of output vector.
    :return: tf.Tensor.
    """
    return linear(x, out_dim, 'linear', with_w=True)#tf.layers.dense(x, out_dim)


def fc_bn_lrelu(x, out_dim, is_train, alpha=0.2):
    """
    Add fc + bn + Leaky Relu layers
    see fc_layer and batchNormalization function
    """
    fc = fc_layer(x, out_dim)
    bn = batchNormalization(fc, is_train)
    return tf.nn.leaky_relu(bn, alpha)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias#, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias