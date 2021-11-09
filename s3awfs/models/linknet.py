import os.path
from functools import lru_cache
import logging

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Add,
    Conv2DTranspose,
    Dropout
)
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from utilitys import ProcessIO


def focal_tversky_loss(Y_true, Y_predicted, gamma = 0.75):
    ti = tversky_index(Y_true, Y_predicted)
    ftl = K.pow((1 - ti), gamma)
    return ftl

def tversky_index(Y_true, Y_predicted, alpha = 0.7):
    Y_true = K.cast(Y_true, K.floatx())
    Y_true = K.flatten(Y_true)
    Y_predicted = K.flatten(Y_predicted)
    true_positive = K.sum(Y_true * Y_predicted)
    false_negative = K.sum(Y_true * (1 - Y_predicted))
    false_positive = K.sum((1 - Y_true) * Y_predicted)
    ti = (true_positive + 1.0) / (true_positive + alpha * false_negative + (1 - alpha) * false_positive)
    return ti

def dice_coefficient(Y_true, Y_predicted, smoothness=1.0):
    Y_true = K.cast(Y_true, K.floatx())
    Y_true = K.flatten(Y_true)
    Y_predicted = K.flatten(Y_predicted)
    Y_predicted = K.cast(Y_predicted, K.floatx())
    intersection = K.sum(Y_true * Y_predicted)
    dc = (2.0 * intersection + smoothness) / (K.sum(Y_true) + K.sum(Y_predicted) + smoothness)
    return dc

customObjects = {
    'focal_tversky_loss': focal_tversky_loss,
    'dice_coefficient': dice_coefficient
}

class LinkNet:
    """
    Class for the LinkNet Semantic Segmentation Neural Network
    """
    def __init__(self, inputShape: tuple, numClasses=2):
        """
        :param inputShape: Shape of the images processed by the network (w, h, nchans)
        :param numClasses: The number of output classes for the Neural Network data.
        """
        self.input_tuple = inputShape
        self.input_layer = Input(self.input_tuple)

        self.conv1 = Conv2D(
            filters=64, kernel_size=(7, 7), strides=2, kernel_initializer="random_normal", padding="same"
        )(self.input_layer)
        self.conv1 = BatchNormalization()(self.conv1)
        self.conv1 = Activation("relu")(self.conv1)
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2)(self.conv1)

        self.encoder1 = self._getEncoderBlock(self.pool1, 64)
        self.encoder2 = self._getEncoderBlock(self.encoder1, 128)
        self.encoder3 = self._getEncoderBlock(self.encoder2, 256)
        self.encoder4 = self._getEncoderBlock(self.encoder3, 512)

        self.decoder4 = self._getDecoderBlock(self.encoder4, 512, 256)
        self.decoder4 = Add()([self.decoder4, self.encoder3])

        self.decoder3 = self._getDecoderBlock(self.decoder4, 256, 128)
        self.decoder3 = Add()([self.decoder3, self.encoder2])

        self.decoder2 = self._getDecoderBlock(self.decoder3, 128, 64)
        self.decoder2 = Add()([self.decoder2, self.encoder1])

        self.decoder1 = self._getDecoderBlock(self.decoder2, 64, 64)

        self.output_layer = Conv2DTranspose(
            filters=32, kernel_size=(3, 3), strides=2, kernel_initializer="random_normal", padding="same"
        )(self.decoder1)
        self.output_layer = BatchNormalization()(self.output_layer)
        self.output_layer = Activation("relu")(self.output_layer)
        self.output_layer = Dropout(0.5)(self.output_layer)

        self.output_layer = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer="random_normal", padding="same")(
            self.output_layer
        )
        self.output_layer = BatchNormalization()(self.output_layer)
        self.output_layer = Activation("relu")(self.output_layer)
        self.output_layer = Dropout(0.5)(self.output_layer)

        self.output_layer = Conv2DTranspose(
            filters=numClasses, kernel_size=(2, 2), strides=2, kernel_initializer="random_normal", padding="same"
        )(self.output_layer)
        self.output_layer = Activation("softmax")(self.output_layer)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer, name="LinkNet")

    def _getEncoderBlock(self, input_layer, output_filters):
        """
        Encoder Block stucture of the model.
        :param input_layer: Layer object to pass through block.
        :param output_filters: Number of filters should be contained at the output of block.
        :return output_layer: Output layer of the encoder block.
        """
        output_layer = Conv2D(
            filters=output_filters, kernel_size=(3, 3), strides=2, kernel_initializer="random_normal", padding="same"
        )(input_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation("relu")(output_layer)

        output_layer = Conv2D(
            filters=output_filters, kernel_size=(3, 3), kernel_initializer="random_normal", padding="same"
        )(output_layer)
        output_layer = BatchNormalization()(output_layer)
        added_layer_1 = self._encoderAddLayers(input_layer, output_layer)
        output_layer = Activation("relu")(added_layer_1)

        output_layer = Conv2D(
            filters=output_filters, kernel_size=(3, 3), kernel_initializer="random_normal", padding="same"
        )(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation("relu")(output_layer)

        output_layer = Conv2D(
            filters=output_filters, kernel_size=(3, 3), kernel_initializer="random_normal", padding="same"
        )(output_layer)
        output_layer = BatchNormalization()(output_layer)
        added_layer_2 = self._encoderAddLayers(added_layer_1, output_layer)
        output_layer = Activation("relu")(added_layer_2)

        return output_layer

    @classmethod
    def _getDecoderBlock(cls, input_layer, input_filters, output_filters):
        """
        Decoder Block stucture of the model.
        :param input_layer: Layer object to pass through block.
        :param input_filters: Number of filters to pass in to the input of the block.
        :param output_filters: Number of filters should be contained at the output of block.
        """
        output_layer = Conv2D(
            filters=int(input_filters / 4), kernel_size=(1, 1), kernel_initializer="random_normal", padding="same"
        )(input_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation("relu")(output_layer)

        output_layer = Conv2DTranspose(
            filters=int(input_filters / 4),
            kernel_size=(3, 3),
            strides=2,
            kernel_initializer="random_normal",
            padding="same",
        )(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation("relu")(output_layer)

        output_layer = Conv2D(
            filters=output_filters, kernel_size=(1, 1), kernel_initializer="random_normal", padding="same"
        )(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation("relu")(output_layer)

        return output_layer

    @classmethod
    def _encoderAddLayers(cls, layer_1, layer_2):
        """
        Helper function for the Encoder Block stucture that adds two input layers.
        :param layer_1: First layer to be added.
        :param layer_2: Second layer to be added.
        :return output_layer: The added layer of the two inputted layers.
        """
        layer_1_shape = layer_1.shape
        layer_2_shape = layer_2.shape
        stride_x = int(round(layer_1_shape[1] / layer_2_shape[1]))
        stride_y = int(round(layer_1_shape[2] / layer_2_shape[2]))
        equal_channels = layer_1_shape[3] == layer_2_shape[3]
        if stride_x > 1 or stride_y > 1 or not equal_channels:
            layer_1 = Conv2D(
                filters=layer_2_shape[3],
                kernel_size=(1, 1),
                strides=(stride_x, stride_y),
                kernel_initializer="random_normal",
            )(layer_1)
        return Add()([layer_1, layer_2])

def makeLinknetModel(
  numClasses=2,
  imageShape=(512,512,3),
  weightsFile=None,
  outputFile=None,
  strategy=None
):
    """
    Creates a h5 model file for the specified input layer topology. Optionally, a weights file can be specified
    to preset layer weights
    """
    strategy = strategy or tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = LinkNet(imageShape, numClasses).model
        meanIou = MeanIoU(num_classes=numClasses)
        metrics = ["accuracy", meanIou]
        model.compile(loss=focal_tversky_loss, metrics=metrics)
    if weightsFile is not None:
        model.load_weights(weightsFile)
    if outputFile is not None and os.path.isfile(outputFile):
        model.save(outputFile)
        model = outputFile
        strategy = None
    return ProcessIO(model=model, strategy=strategy, customObjects=customObjects)

@lru_cache()
def loadLinknetModel(file, strategy=None):
    """
    :param file: h5 model file to load
    type: filepicker
    value: ''
    fileFilter: Tensorflow Model Files (*.h5);;
    :param strategy: Tensorflow strategy, can be set with ``makeTensorflowStrategy``
    ignore: True
    """
    strategy = strategy or tf.distribute.MirroredStrategy()
    with strategy.scope():
       model = load_model(file, custom_objects=customObjects)
    return ProcessIO(strategy=strategy, model=model)