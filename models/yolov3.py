import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Concatenate,
    Input,
    ZeroPadding2D,
    BatchNormalization,
    LeakyReLU,
    Lambda,
    UpSampling2D,
)
from tensorflow.keras.regularizers import l2

# Anchors from arXiv:1804.02767. They are derived using k-means clustering on COCO data set.
yolo_anchors = (
    np.array(
        [
            (10, 13),
            (16, 30),
            (33, 23),
            (30, 61),
            (62, 45),
            (59, 119),
            (116, 90),
            (156, 198),
            (373, 326),
        ],
        np.float32,
    )
    / 416
)

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = "same"
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # Top left half-padding.
        padding = "valid"

    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=l2(0.0005),
    )(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])

    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)

    return x


def Darknet53(name=None):
    """
    Darknet-53 backbone as described in arXiv:1804.02767.
    """
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)
    x = x_36 = DarknetBlock(x, 256, 8)
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)

    return Model(inputs, (x_36, x_61, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # Concatenate with skip connection.
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, 2 * filters, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, 2 * filters, 3)
        x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, 2 * filters, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(
            lambda x: tf.reshape(
                x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)
            )
        )(x)
        return Model(inputs, x, name=name)(x_in)

    return yolo_output


def YOLOv3(
    width=None,
    height=None,
    channels=3,
    anchors=yolo_anchors,
    masks=yolo_anchor_masks,
    classes=80,
    training=False,
):
    x = inputs = Input([height, width, channels], name="input")

    x_36, x_61, x = Darknet53(name="darknet-53")(x)

    x = YoloConv(512, name="yolo_conv_0")(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name="yolo_output_0")(x)

    x = YoloConv(256, name="yolo_conv_1")((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name="yolo_output_1")(x)

    x = YoloConv(128, name="yolo_conv_2")((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name="yolo_output_2")(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name="yolov3")
