import sys
sys.path.insert(0, './..')


from layers import encoder, decoder, sample_z, kl, compute_mmd
import os
from data import Circle
import numpy as np
import tensorflow as tf
import h5py


def train(size, 
          radius,
          batch_size,
          beta,
          epochs,
          snapshot_dir,
          checkpoint_dir,
          regularizer="kl",
          z_dim=2):

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    in_height = size
    in_width = size

    circle = Circle(size, 
                    radius)


    y = tf.placeholder(tf.float32, shape=[batch_size, 
                                          in_height, 
                                          in_width, 
                                          1])
    print("Y ENCODER:")
    latents = encoder(y, 2*z_dim, scope="y_encoder")
    print("latents: " + str(latents.get_shape().as_list()))

    mean = latents[:,:z_dim]
    log_sigma = latents[:,z_dim:]
    z = sample_z(mean, log_sigma)

    y_logits = decoder(z)
    print("y_logits: " + str(y_logits.get_shape().as_list()))
    y_out = tf.sigmoid(y_logits)

    # Test time:
    z_test = tf.placeholder(tf.float32, shape=[None, z_dim])
    y_test = tf.sigmoid(decoder(z_test, reuse=True))


    kl_loss = kl(mean, log_sigma, batch_size)

    true_samples = tf.random_normal(tf.stack([batch_size, z_dim]))
    mmd_loss = compute_mmd(true_samples, z)

    if regularizer == "mmd":
        reg_loss = mmd_loss
    else:
        reg_loss = kl_loss

    ce = tf.losses.sigmoid_cross_entropy(y,
                                         y_logits)
    
    loss = beta * reg_loss + ce
    opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(epochs):
        batch_x, batch_y = circle.next_batch(batch_size)
        batch_y = batch_y.reshape(batch_size, in_height, in_width, 1)

        _, ce_i, kl_i, z_i, y_out_i, mmd_i =\
                sess.run([opt, ce, kl_loss, z, y_out, mmd_loss], feed_dict={y: batch_y})

        if i % 100 == 0:
            print("iteration {} - ce: {}, kl: {}, mmd: {}".format(i, ce_i, kl_i, mmd_i))

        if i % 1000 == 0:
            snapshot_path = os.path.join(snapshot_dir, "snap_{}.h5".format(i))
            with h5py.File(snapshot_path, 'w') as f:
                f.create_dataset(name="y",
                                 data=batch_y[:,:,:,0],
                                 compression='gzip')

                f.create_dataset(name="y_out",
                                 data=y_out_i[:,:,:,0],
                                 compression='gzip')

                f.create_dataset(name="z",
                                 data=z_i,
                                 compression='gzip')

                y_test_i, z_test_i = sess.run([y_test, z_test], 
                                     feed_dict={z_test: np.random.normal(size=(batch_size, z_dim))})

                f.create_dataset(name="y_test",
                                 data=y_test_i[:,:,:,0],
                                 compression="gzip")

                f.create_dataset(name="z_test",
                                 data=z_test_i,
                                 compression="gzip")


        if i % 10000 == 0:
            saver.save(sess, os.path.join(checkpoint_dir, "vae"), global_step=i)
            

    sess.close()

if __name__ == "__main__":
    train(size=28, 
          radius=12,
          batch_size=200,
          beta=100.0,
          epochs=100000,
          snapshot_dir="./snapshots/run_14",
          checkpoint_dir="./checkpoints/run_14",
          regularizer="mmd",
          z_dim=2)


