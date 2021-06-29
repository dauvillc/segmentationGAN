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
from segmentationGAN.models import get_patch_discriminator, get_unet_architecture
from segmentationGAN.preprocessing import coshuffle_arrays, coshuffle_lists
from segmentationGAN.optim import LinearLRSchedule


class SegmentationGAN:
    """
    A Segmentation GAN is a model adapted from
    P.Isola et al. )s pix2pix for segmentation tasks.
    """
    def __init__(self, name, generator, input_shape, nb_classes, use_input_for_discriminator=True,
                 use_correction=False):
        """
        -- name: Name of this model
        -- generator: tf.keras.Model to be used as generator
        -- input_shape: shape of the input images
        -- nb_classes: Number of classes in this segmentation task.
        -- use_use_input_for_discriminator: Whether the discriminator should receive the input image
                    when classifying the segmentations as real or fake.
        -- use_correction: whether to use a second network which learns to correct the images
                           of the fully trained generator.
        """
        self._gen = generator
        self._disc = get_patch_discriminator(input_shape, nb_classes, use_input_for_discriminator)

        self._name = name
        self._nb_classes = nb_classes
        self._use_input_for_disc = use_input_for_discriminator

        # correction network parameters
        self._use_correction = use_correction
        self._correction_network = get_unet_architecture(input_shape[0] + nb_classes, input_shape[1],
                                                         input_shape[1], nb_classes, style="gan")
        tf.keras.utils.plot_model(self._correction_network, to_file="correc_model.jpg", show_shapes=True)
        self._corr_training_prop = 0.2  # Proportion of the training images kept to train the second network

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

        # If a correction network is used, isolate a part of the training data
        # which will be used to train the correction network
        if self._use_correction:
            nb_corr_cases = int(self._corr_training_prop * inputs.shape[0])
            correction_inputs, correction_targets = inputs[:nb_corr_cases], targets[:nb_corr_cases]
            inputs, targets = inputs[:nb_corr_cases], targets[:nb_corr_cases]
            print("Isolating {} cases for correction network training".format(nb_corr_cases))

        # Split the data into batches
        total_batches = int(np.ceil(inputs.shape[0] // batch_size))
        input_batches = np.array_split(inputs, total_batches)
        target_batches = np.array_split(targets, total_batches)

        # Optimizers and Hyperparameters
        gen_lr_schedule = LinearLRSchedule(2e-4, total_batches * epochs)
        disc_lr_schedule = LinearLRSchedule(2e-4, total_batches * epochs)

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

            # Save the model every 20 epochs
            if n_epoch % 20 == 0:
                print("Saving model into ", savefile)
                self._gen.save(savefile)

        # Save the model at the end of the training phase
        print("Saving model into ", savefile)
        self._gen.save(savefile)

        # ===========================================================
        # CORRECTION NETWORK TRAINING (IF NEEDED)
        # We will now use the generator to segment the images which were kept for correction training,
        # and pass them to the correction network. The targets remain the same, and the
        # correction network also receives the input images.
        if self._use_correction:
            segmentations = self._gen(correction_inputs)

            # Stacks the generated segmentations and the original images along the channels axis
            # If the images are RGB and there are 2 segmentation classes, the shape of
            # correction_data will be (batch_size, 3 + 2, h, w)
            correction_data = np.stack((segmentations, correction_inputs), axis=1)

            correction_lr = LinearLRSchedule(2e-4, epochs//2 * total_batches)
            correction_optim = tf.keras.optimizers.Adam(learning_rate=correction_lr, beta_1=0.5)
            self._correction_network.compile(optimizer=correction_optim, loss=losses.correction_loss)

            self._correction_network.fit(correction_data, correction_targets, batch_size=batch_size, epochs=epochs//2)

        # ===========================================================

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
