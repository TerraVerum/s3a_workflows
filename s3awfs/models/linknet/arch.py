from tensorflow.keras.layers import *
from tensorflow.keras.models import *

# Model Implementation based on the following
# https://arxiv.org/pdf/1707.03718.pdf
# https://github.com/e-lab/LinkNet


class LinkNet:
    """
    Class for the LinkNet Semantic Segmentation Neural Network
    """

    def __init__(self, height, width, output_classes, input_channels=3):
        """
        :param height: Height of the Image
        :param width: Width of the Image
        :param output_classes: The number of output classes for the Neural Network data.
        :param input_channels: Channels of the Image
        """
        self.input_tuple = (height, width, input_channels)
        self.input_layer = Input(self.input_tuple)

        self.conv1 = Conv2D(
            filters=64, kernel_size=(7, 7), strides=2, kernel_initializer="random_normal", padding="same"
        )(self.input_layer)
        self.conv1 = BatchNormalization()(self.conv1)
        self.conv1 = Activation("relu")(self.conv1)
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2)(self.conv1)

        self.encoder1 = self.__get_encoder_block(self.pool1, 64)
        self.encoder2 = self.__get_encoder_block(self.encoder1, 128)
        self.encoder3 = self.__get_encoder_block(self.encoder2, 256)
        self.encoder4 = self.__get_encoder_block(self.encoder3, 512)

        self.decoder4 = self.__get_decoder_block(self.encoder4, 512, 256)
        self.decoder4 = Add()([self.decoder4, self.encoder3])

        self.decoder3 = self.__get_decoder_block(self.decoder4, 256, 128)
        self.decoder3 = Add()([self.decoder3, self.encoder2])

        self.decoder2 = self.__get_decoder_block(self.decoder3, 128, 64)
        self.decoder2 = Add()([self.decoder2, self.encoder1])

        self.decoder1 = self.__get_decoder_block(self.decoder2, 64, 64)

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
            filters=output_classes, kernel_size=(2, 2), strides=2, kernel_initializer="random_normal", padding="same"
        )(self.output_layer)
        self.output_layer = Activation("softmax")(self.output_layer)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer, name="LinkNet")

    def __get_encoder_block(self, input_layer, output_filters):
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
        added_layer_1 = self.__encoder_add_layers(input_layer, output_layer)
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
        added_layer_2 = self.__encoder_add_layers(added_layer_1, output_layer)
        output_layer = Activation("relu")(added_layer_2)

        return output_layer

    def __get_decoder_block(self, input_layer, input_filters, output_filters):
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

    def __encoder_add_layers(self, layer_1, layer_2):
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

    def get_model(self):
        """
        Returns the model object
        """
        return self.model
