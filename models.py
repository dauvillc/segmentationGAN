"""
Defines models creation and manipulation functions.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers


def get_unet_architecture(n_ch,
                          input_height,
                          input_width,
                          nb_classes,
                          style="classic"):
    """
    Defines a UNet architecture of adapted to the given image size.
    -- n_ch: Number of channels to be expected in the images;
    -- input_height: Input images height;
    -- input_width: Input images width.
    -- nb_classes: Number of segmentation classes
    -- style: 'classic' or 'gan'. If GAN, no reshaping is applied.
    """
    # Since the unet contains 3 max pooling layers, we need to make
    # the input dimensions divided by 8
    input_height = int(input_height / 8) * 8
    input_width = int(input_width / 8) * 8
    print("Input shape: ", input_height, input_width)

    inputs = layers.Input(shape=(n_ch, input_height, input_width))

    conv1 = layers.Conv2D(32, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(inputs)
    conv1 = layers.Dropout(0.3)(conv1)
    conv2 = layers.Conv2D(32, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(conv1)
    pool1 = layers.MaxPooling2D((2, 2), data_format='channels_first')(conv2)

    conv3 = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(pool1)
    conv3 = layers.Dropout(0.3)(conv3)
    conv4 = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(conv3)
    pool2 = layers.MaxPooling2D((2, 2), data_format='channels_first')(conv4)

    conv5 = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(pool2)
    conv5 = layers.Dropout(0.3)(conv5)
    conv6 = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(conv5)
    pool3 = layers.MaxPooling2D((2, 2), data_format='channels_first')(conv6)

    conv7 = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(pool3)
    conv7 = layers.Dropout(0.3)(conv7)
    conv8 = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(conv7)

    up1 = layers.UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up1 = layers.concatenate([conv6, up1], axis=1)
    conv9 = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          data_format='channels_first')(up1)
    conv9 = layers.Dropout(0.3)(conv9)
    conv10 = layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same',
                           data_format='channels_first')(conv9)

    up2 = layers.UpSampling2D(size=(2, 2),
                              data_format='channels_first')(conv10)
    up2 = layers.concatenate([conv4, up2], axis=1)
    conv11 = layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           data_format='channels_first')(up2)
    conv11 = layers.Dropout(0.3)(conv11)
    conv12 = layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           data_format='channels_first')(conv11)

    up3 = layers.UpSampling2D(size=(2, 2),
                              data_format='channels_first')(conv12)
    up3 = layers.concatenate([conv2, up3], axis=1)
    conv13 = layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same',
                           data_format='channels_first')(up3)
    conv13 = layers.Dropout(0.3)(conv13)
    conv14 = layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same',
                           data_format='channels_first')(conv13)

    otpt = None
    if style == "gan":
        otpt = layers.Conv2D(
            nb_classes,
            (1, 1),
            padding='same',
            data_format='channels_first',
        )(conv14)
        otpt = tf.nn.softmax(otpt, axis=1)
    else:
        conv15 = layers.Conv2D(3, (1, 1),
                               activation='relu',
                               padding='same',
                               data_format='channels_first')(conv14)
        otpt = layers.Reshape((nb_classes, input_height * input_width))(conv15)
        otpt = layers.Permute((2, 1))(otpt)

        otpt = layers.Activation('softmax')(otpt)

    model = Model(inputs=inputs, outputs=otpt)

    return model


def downsampling(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters,
                      size,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      data_format="channels_first"))
    result.add(
        layers.Conv2D(filters,
                      3,
                      strides=1,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      data_format="channels_first"))

    if apply_batchnorm:
        result.add(layers.BatchNormalization(axis=1))

    result.add(tf.keras.layers.LeakyReLU())

    return result


def get_patch_discriminator(input_image_shape,
                            nb_classes,
                            use_input_image=True):
    """
    Returns a patch net discriminator.
    -- input_image_shape: Shape of the input image, including the channels
    -- nb_classes: Number of classes in the segmentation task
    -- use_input_image: if False, the discriminator will not receive the input image to classify
                        the segmentation as real or fake.
    A Patch net is a fully convolutional network which classifies
    each pixels into two possible classes (real / fake in the case of
    a GAN).
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    image_input = layers.Input(shape=input_image_shape, name="input_img")

    x = layers.Input(
        shape=[nb_classes, input_image_shape[1], input_image_shape[2]],
        name="target_image")
    inputs = [x]

    if use_input_image:
        inputs = [image_input, x]
        x = layers.concatenate(inputs, axis=1)  # (bs, 96, 96, 5)

    down1 = downsampling(64, 4, False)(x)  # (bs, 48, 48, 64)
    down2 = downsampling(128, 4)(down1)  # (bs, 24, 24, 128)
    down3 = downsampling(256, 4)(down2)  # (bs, 12, 12, 256)

    zero_pad1 = layers.ZeroPadding2D(data_format="channels_first")(
        down3)  # (bs, 14, 14, 256)
    conv = layers.Conv2D(512,
                         4,
                         strides=1,
                         kernel_initializer=initializer,
                         use_bias=False,
                         data_format="channels_first")(
                             zero_pad1)  # (bs, 11, 11, 512)
    batchnorm1 = layers.BatchNormalization(axis=1)(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D(data_format="channels_first")(
        leaky_relu)  # (bs, 13, 13,256)
    last = layers.Conv2D(1,
                         4,
                         strides=1,
                         data_format="channels_first",
                         kernel_initializer=initializer)(
                             zero_pad2)  # (bs,10,10,256)

    return Model(inputs=inputs, outputs=last)
