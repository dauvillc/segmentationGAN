"""
Defines loss functions, including the generator loss
for the pix2pix model.

Most code (including the GAN losses) is taken from the tensorflow Core
pix2pix example.
"""

import tensorflow as tf
import numpy as np

bin_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mae = tf.keras.losses.MeanAbsoluteError()


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of the categorical cross entropy.
    Target tensors must be encoded in one-hot.
    """

    weights = tf.Variable(weights, dtype=tf.float32)
    epsilon = tf.keras.backend.epsilon()

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1)

        # Sums over the spatial dimensions - output: (bs, 3)
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=(2, 3))

        # Average over the batch - output: (3,)
        loss = tf.reduce_mean(loss, axis=0)

        # Applies the weights
        loss = loss * weights

        return tf.reduce_mean(loss)

    return loss


def generator_loss(disc_generated_output, gen_output, target, base_loss_func):
    """
    Loss for the generator network.
    -- disc_generated_output: Discriminator output on the generated images.
                              Should be given as logits, not probabilities.
    -- gen_output: Generator output (Discriminator's input)
    -- target: groundtruth data
    -- base_loss_func: Function to use as base loss (crossentropy,
                       L1 distance, ...).
    returns: T, G, bc where T is the total loss, g is the gan loss term,
    and bc is the binary crossentropy term.
    """
    # We compare the disc's output on fake images to an array of ones
    # because the generator wants the disc to output ones on fake images
    gan_loss = bin_crossentropy(tf.ones_like(disc_generated_output),
                                disc_generated_output)
    base_loss = base_loss_func(target, gen_output)

    # the final loss mixes both terms, with the default lambda coeff of 10
    total_gen_loss = base_loss + gan_loss

    return total_gen_loss, gan_loss, base_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Loss for the discriminator network.
    -- disc_real_output: Output of the disc. on real images.
    -- disc_generated_output: Output of the disc on fake images.
    returns: the discriminator loss.
    """
    real_loss = bin_crossentropy(tf.ones_like(disc_real_output),
                                 disc_real_output)
    generated_loss = bin_crossentropy(tf.zeros_like(disc_generated_output),
                                      disc_generated_output)
    return real_loss + generated_loss
