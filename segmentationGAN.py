"""
Defines the SegmentationGAN class
"""

import os
import tensorflow as tf
import numpy as np
import time
import segmentationGAN.losses as losses
from skimage.io import imsave
from segmentationGAN.image import convert_to_8bits_rgb
from segmentationGAN.models import get_patch_discriminator
from segmentationGAN.preprocessing import coshuffle_arrays, coshuffle_lists


class SegmentationGAN:
    """
    A Segmentation GAN is a model adapted from
    P.Isola et al. )s pix2pix for segmentation tasks.
    """
    def __init__(self, name, generator, input_shape, nb_classes, use_input_for_discriminator=True):
        """
        -- name: Name of this model
        -- generator: tf.keras.Model to be used as generator
        -- nb_classes: Number of classes in this segmentation task.
        -- use_use_input_for_discriminator: Whether the discriminator should receive the input image
                    when classifying the segmentations as real or fake.
        """
        self._gen = generator
        self._disc = get_patch_discriminator(input_shape, nb_classes, use_input_for_discriminator)

        self._name = name
        self._nb_classes = nb_classes
        self._use_input_for_disc = use_input_for_discriminator

    def train(self,
              inputs,
              targets,
              epochs=1,
              batch_size=32,
              validation_data=None,
              checkpoint_dir="./",
              class_weights=None):
        """
        Trains the segmentation GAN.
        -- inputs: array of shape (samples, c, h, w)
        -- targets: array of shape (samples, h, w).
                    The target must be encoded sparsely, i.e.
                    target[i, j, k] = c where c is the index of one
                    of the segmentation classes.
        -- epochs: Number of epochs of training
        -- batch_size: Well, that's the batch size. The higher the batch size,
                       the more GPU memory-consuming.
        -- validation_data: optional couple (valid_inputs, valid_targets)
                            to evaluate the network after training.
        -- class_weights: Array indicating the weight for each segmentation
                          class. If None, all classes will be of equal
                          importance in the loss.
        """
        if class_weights is not None:
            weighted_loss = losses.weighted_categorical_crossentropy(
                class_weights)
        else:
            # If no weights were passed, consider them to be all one
            weighted_loss = losses.weighted_categorical_crossentropy(
                np.ones((self._nb_classes, )))

        # Shuffles the data before starting the training
        inputs, targets = coshuffle_arrays(inputs, targets)

        # First, the targets are converted to one-hot encoding
        targets = tf.one_hot(targets.astype(np.uint8),
                             self._nb_classes,
                             axis=1)

        # Split the data into batches
        total_batches = int(np.ceil(inputs.shape[0] // batch_size))
        input_batches = np.array_split(inputs, total_batches)
        target_batches = np.array_split(targets, total_batches)

        # Optimizers and Hyperparameters
        gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=total_batches, decay_rate=0.97)

        disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=total_batches, decay_rate=0.97)
        gen_optim = tf.keras.optimizers.Adam(learning_rate=gen_lr_schedule,
                                             beta_1=0.5)
        disc_optim = tf.keras.optimizers.Adam(learning_rate=disc_lr_schedule,
                                              beta_1=0.5)

        # Make sure the checkpoints directory actually exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        savefile = os.path.join(checkpoint_dir, self._name + "_best_weights.h5")

        # TRAINING LOOP ============================================

        for n_epoch in range(epochs):
            epoch_start_time = time.time()

            input_batches, target_batches = coshuffle_lists(
                input_batches, target_batches)

            for n_batch, (inpt, target) in enumerate(
                    zip(input_batches, target_batches)):

                gen_loss, gan_loss, base_loss, disc_loss = self.train_step(
                    inpt, target, weighted_loss, gen_optim, disc_optim)
                print("Batch {} /{}      ".format(n_batch, total_batches),
                      end="\r")

            print("Epoch {} ended in {:1.3f}s".format(
                n_epoch + 1,
                time.time() - epoch_start_time))
            print(
                "GEN base loss: {:1.3f} gan loss: {:1.3f} total loss: {:1.3f} \
DISC loss: {:1.3f}".format(base_loss, gan_loss, gen_loss, disc_loss))
            print("Current LR: ",
                  gen_optim.learning_rate((n_epoch + 1) * batch_size))

            # Save the model every 20 epochs
            if n_epoch % 20 == 0:
                print("Saving model into ", savefile)
                self._gen.save(savefile)

        # ===========================================================

        # Save the model at the end of the training phase
        print("Saving model into ", savefile)
        self._gen.save(savefile)

        # Applies the gen to the validation data
        print("Applying to the validation data")
        if validation_data is not None:
            valid_inpt, valid_target = validation_data
            self.save_validation_examples(valid_inpt, valid_target, checkpoint_dir)

    @tf.function
    def train_step(self, input_image, target, loss, gen_optim, disc_optim):
        """
        Performs a single training step over a batch of data.
        -- input_image: Input data
        -- target: Target data
        -- loss: Loss function to use as base loss (could be crossentropy,
                 L1 distance, etc..).
        -- gen_optim: Optimizer object for the generator
        -- disc_optim: Optimizer object for the discriminator
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass
            gen_output = self._gen(input_image, training=True)

            disc_real_inputs = [target]
            if self._use_input_for_disc:
                disc_real_inputs = [input_image, target]
            disc_real_output = self._disc(disc_real_inputs, training=True)

            # Check whether the discriminator receives the input
            disc_fake_inputs = [gen_output]
            if self._use_input_for_disc:
                disc_fake_inputs = [input_image, gen_output]

            disc_generated_output = self._disc(disc_fake_inputs,
                                               training=True)

            # Compute the losses
            gen_loss, gen_gan_loss, gen_base_loss = losses.generator_loss(
                disc_generated_output, gen_output, target, loss)
            disc_loss = losses.discriminator_loss(disc_real_output,
                                                  disc_generated_output)

            # Compute the gradients
            gen_gradients = gen_tape.gradient(gen_loss,
                                              self._gen.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss,
                                                self._disc.trainable_variables)

            # Optimize the models' parameters
            gen_optim.apply_gradients(
                zip(gen_gradients, self._gen.trainable_variables))
            disc_optim.apply_gradients(
                zip(disc_gradients, self._disc.trainable_variables))

        return gen_loss, gen_gan_loss, gen_base_loss, disc_loss

    def save_validation_examples(self,
                                 inputs,
                                 targets,
                                 results_dir="seggan_valid"):
        """
        Evaluates the model onto examples and saves the segmentations
        as images.
        """
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        predictions = self._gen(inputs)
        segmentations = np.argmax(predictions, axis=1)

        for i in range(inputs.shape[0]):
            mask = convert_to_8bits_rgb(targets[i])
            img = convert_to_8bits_rgb(inputs[i])
            seg = convert_to_8bits_rgb(segmentations[i])
            imsave(os.path.join(results_dir, "{}_mask.jpg".format(i)),
                   mask,
                   check_contrast=False)
            imsave(os.path.join(results_dir, "{}_img.jpg".format(i)), img)
            imsave(os.path.join(results_dir, "{}_seg.jpg".format(i)),
                   seg,
                   check_contrast=False)
        return

    def segment(self, x, results_dir="seggan_test", batch_size=1):
        """
        Performs segmentation on a given set of data.
        -- x: Input data, as an array of shape (batch_size, c, h, w).
        returns the segmentations as an array of shape (batch_size, h, w)
        """
        probas = self.__gen.predict(x, batch_size=batch_size)
        return np.argmax(probas, axis=1)
