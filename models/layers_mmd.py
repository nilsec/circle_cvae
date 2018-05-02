import tensorflow as tf
import numpy as np

def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)

def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(
                                        inputs, 
                                        num_outputs, 
                                        kernel_size, 
                                        stride,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation_fn=tf.identity)
    conv = lrelu(conv)
    return conv

def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv

def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)

    fc = lrelu(fc)
    return fc

def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc

def fc_id(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)

    return fc

def sample_z(mean,
             log_sigma):

    dim = mean.get_shape().as_list()
    normal = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(dim), tf.ones(dim))
    sample = normal.sample(sample_shape=1)

    z_sigma = tf.multiply(tf.exp(log_sigma), sample)
    z = tf.add(mean, z_sigma)

    return z[0,:,:]


def kl(mean, log_sigma, batch_size, free_bits=0.0):
    kl_div = tf.reduce_sum(tf.maximum(free_bits,
                                      0.5 * (tf.square(mean) + tf.exp(2 * log_sigma) - 2 * log_sigma - 1)))
    kl_div /= float(batch_size)
    return kl_div


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


# Encoder and decoder use the DC-GAN architecture
def encoder(x, z_dim, scope="encoder"):
    print("ENCODER:")
    with tf.variable_scope(scope):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        print(" conv1: " + str(conv1.get_shape().as_list()))
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        print(" conv2: " + str(conv2.get_shape().as_list()))
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        print(" reshape: " + str(conv2.get_shape().as_list()))
        fc1 = fc_lrelu(conv2, 1024)
        print(" fc1: " + str(fc1.get_shape().as_list()))
        fc2 = fc_id(fc1, z_dim)
        print(" fc2: " + str(fc2.get_shape().as_list()))
        return fc2


def decoder(z):
    print("DECODER")
    with tf.variable_scope('decoder'):
        fc1 = fc_relu(z, 1024)
        print(" fc1: " + str(fc1.get_shape().as_list()))
        fc2 = fc_relu(fc1, 7*7*128)
        print(" fc2: " + str(fc2.get_shape().as_list()))
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        print(" reshape: " + str(fc2.get_shape().as_list()))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        print(" conv1t: " + str(conv1.get_shape().as_list()))
        output = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.identity)
        print(" conv2t: " + str(output.get_shape().as_list())) 
        return output
