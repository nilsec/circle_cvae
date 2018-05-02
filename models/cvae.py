from layers_mmd import encoder, decoder, sample_z, kl
import numpy as np
import tensorflow as tf


def train(in_height, 
          in_width,
          batch_size,
          z_dim=20):

    x = tf.placeholder(tf.float32, shape=[batch_size, 
                                          in_height, 
                                          in_width, 
                                          1])

    y = tf.placeholder(tf.float32, shape=[batch_size, 
                                          in_height, 
                                          in_width, 
                                          1])
    xy = tf.concat([x,y], axis=-1)

    latents = encoder(xy, 2*z_dim, scope="xy_encoder")

    mean = latents[:,:z_dim]
    log_sigma = latents[:,z_dim:]

    z = sample_z(mean, log_sigma)
    x = encoder(x, z_dim, scope="x_encoder")

    xz = tf.concat([x,z], axis=-1)

    y_logits = decoder(xz)
    y_out = tf.sigmoid(y_logits)

if __name__ == "__main__":
    train(28, 
          28,
          20,
          20)

