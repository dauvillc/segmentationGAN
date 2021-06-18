"""
Defines the SegmentationGAN class
"""

import os
import tensorflow as tf
import numpy as np
import segmentationGAN.losses as losses
from skimage.io import imsave
from segmentationGAN.image import convert_to_8bits_rgb

weighted_loss = losses.weighted_categorical_crossentropy(
    np.array([0.4, 1.4, 0.8]))


class SegmentationGAN:
    """
    A Segmentation GAN is a model adapted from
    P.Isola et al. )s pix2pix for segmentation tasks.
    """
    def __init__(self, name, generator, nb_classes):
        """
        -- name: Name of this model
        -- generator: tf.keras.Model to be used as generator
        -- nb_classes: Number of classes in this segmentation task.
        """
        self._gen = generator
        self._name = name
        self._nb_classes = nb_classes

    def train(self,
              inputs,
              targets,
              validation_data=None,
              checkpoint_dir="./"):
        """
        Trains the segmentation GAN.
        -- inputs: array of shape (samples, c, h, w)
        -- targets: array of shape (samples, h, w).
                    The target must be encoded sparsely, i.e.
                    target[i, j, k] = c where c is the index of one
                    of the segmentation classes.
        -- validation_data: optional couple (valid_inputs, valid_targets)
                            to evaluate the network after training.
        """
        # First, the targets are converted to one-hot encoding
        targets = tf.one_hot(targets.astype(np.uint8),
                             self._nb_classes,
                             axis=1)

        self._gen.compile(optimizer="adam", loss=weighted_loss)
        self._gen.fit(inputs, targets, epochs=5, batch_size=32)
        self._gen.save(
            os.path.join(checkpoint_dir, self._name + "_best_weights.h5"))

        if validation_data is not None:
            valid_inpt, valid_target = validation_data
            self.save_validation_examples(valid_inpt, valid_target)

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
