import tensorflow as tf
import numpy as np

def conv2d_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv

def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv

def fc_id(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    return fc

def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc


def sample_z(mean,
             log_sigma):

    print("SampleZ:")
    print(" mean: " + str(mean.get_shape().as_list()))
    print(" log_sigma: " + str(log_sigma.get_shape().as_list()))

    dim = mean.get_shape().as_list()
    normal = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(dim), tf.ones(dim))
    sample = normal.sample(sample_shape=1)

    z_sigma = tf.multiply(tf.exp(log_sigma), sample)
    z = tf.add(mean, z_sigma)

    print(" z: " + str(z.get_shape().as_list()))
    return z[0,:,:]


def kl(mean, log_sigma, batch_size, free_bits=0.0):
    print("KL:")
    print(" mean: " + str(mean.get_shape().as_list()))
    print(" log_sigma: " + str(log_sigma.get_shape().as_list()))

    kl_div = tf.reduce_sum(tf.maximum(free_bits,
                                      0.5 * (tf.square(mean) + tf.exp(2 * log_sigma) - 2 * log_sigma - 1)))
    kl_div /= float(batch_size)
    print(" kl_div: " + str(kl_div.get_shape().as_list()))
    return kl_div


# Encoder and decoder use the DC-GAN architecture
def encoder(x, z_dim, scope="encoder"):
    with tf.variable_scope(scope):
        conv1 = conv2d_relu(x, 64, 4, 2)
        conv2 = conv2d_relu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_id(conv2, 1024)
        fc2 = fc_id(fc1, z_dim)
        return fc2

def decoder(z):
    with tf.variable_scope('decoder'):
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.identity)
        return output
