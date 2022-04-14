from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from s3a import generalutils as gutils
from skimage.measure import regionprops, label
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils.np_utils import to_categorical

from s3awfs.constants import RNG

class SequenceDataGenerator(Sequence):
    """
    A class that serves as a custom data generator for the Neural Network pipeline.
    Inputs different file names, reads in image files and masks in batches based on batch size,
     and returns the batch of data for a specific data type of Train, Validation, or Test.
    """

    def __init__(
        self,
        ownedImageNames: list[Path | str],
        imagesDir: Path,
        labelMasksDir,
        imageShape,
        numOutputClasses,
        batchSize,
        shuffle=True
    ):
        """
        :param ownedImageNames: list
            A list of Path object file directories for the specific data type.
        :param imagesDir: Path
            Images to use as inputs data
        :param labelMasksDir: Path
            Masks to use as ground truth outputs
        :param batchSize: int
            A int of the batch size of the training of the Neural Network.
        :param imageShape: tuple
            A size two tuple of the height and width of the image.
        :param numOutputClasses: Number of total classes present
        :param shuffle: bool
            A boolean that indicates whether the data should be shuffled at the end of each epoch.
        """
        np.random.seed(22)
        names = []
        for fpath in ownedImageNames:
            if isinstance(fpath, Path):
                fpath = fpath.name
            names.append(fpath)
        self.ownedImageNames = names
        self.imagesDir = imagesDir
        self.batchSize = batchSize
        self.imageShape = imageShape
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.ownedImageNames))
        self.numOutputClasses = numOutputClasses
        self.labelMasksDir = labelMasksDir
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches in an epoch of the data.
        """
        return int(np.floor(len(self.ownedImageNames) / self.batchSize))

    def __getitem__(self, index):
        """
        Gets the batch of data with the specified batch size at a specific index of the data.
        """
        start = index * self.batchSize
        end = (index + 1) * self.batchSize
        indexes = self.indexes[start:end]
        image_names_temp = [self.ownedImageNames[i] for i in indexes]
        X, Y = self.__data_generation(image_names_temp)
        return X, Y

    def on_epoch_end(self):
        """
        Gets executed at the end of each epoch and recollects all file ids and shuffles the ids if the shuffle parameter is set.
        """
        self.indexes = np.arange(len(self.ownedImageNames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_names):
        """
        Returns an array of the images and segmentation masks for a batch of data.
        :param image_names: The file ids of the specific batch of data.
        """
        num_classes = self.numOutputClasses
        images = np.empty((len(image_names), *self.imageShape), dtype=np.uint8)
        masks = np.empty((len(image_names), *self.imageShape[:2], num_classes), dtype=np.uint8)
        for index, imageFile in enumerate(image_names):
            img = gutils.cvImreadRgb(self.imagesDir / imageFile, cv.IMREAD_UNCHANGED)
            mask = self.getMask(imageFile)
            images[index, ...] = img
            masks[index, ...] = mask
        return images, masks

    def getMask(self, maskName):
        mask = gutils.cvImreadRgb(self.labelMasksDir / maskName, cv.IMREAD_UNCHANGED)
        mask = to_categorical(mask, num_classes=self.numOutputClasses, dtype=np.uint8)
        return mask

class SquareMaskSequenceDataGenerator(SequenceDataGenerator):
    def getMask(self, maskName):
        mask = gutils.cvImreadRgb(self.labelMasksDir / maskName, cv.IMREAD_UNCHANGED)
        if self.numOutputClasses > 2:
            raise ValueError('Square mask only works with binary initial labels')
        for region in regionprops(label(mask)):
            bbox = region.bbox
            mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
        return to_categorical(mask, num_classes=self.numOutputClasses, dtype=np.uint8)

class DataGenIterator:
    def __init__(
        self,
        ownedImageNames,
        imagesDir: Path,
        labelMasksDir,
        imageShape,
        numOutputClasses,
        shuffle=True,
        **_kwargs
    ):
        np.random.seed(22)
        names = []
        for fpath in ownedImageNames:
            if isinstance(fpath, Path):
                fpath = fpath.name
            names.append(fpath)
        self.ownedImageNames = names
        self.imagesDir = imagesDir
        self.shuffle = shuffle
        self.numOutputClasses = numOutputClasses
        self.imageShape= imageShape
        self.maskShape = (*imageShape[:2], numOutputClasses)
        self.labelMasksDir = labelMasksDir
        if shuffle:
            RNG.shuffle(self.ownedImageNames)

    def process_path(self, file_path):
        rets = []
        for parent, numChannels in zip(
            [self.imagesDir, self.labelMasksDir],
            [self.imageShape[-1], 1]
        ):
            # Convert the compressed string to a 3D uint8 tensor
            img = tf.io.read_file(str(parent/file_path))
            img = tf.io.decode_png(img, channels=numChannels)
            # Resize the image to the desired size
            rets.append(tf.image.resize(img, self.imageShape[:2]))
        # Load the raw data from the file as a string
        return tuple(rets)

    def __getitem__(self, index):
        """
        Gets the batch of data with the specified batch size at a specific index of the data.
        """
        name = self.ownedImageNames[index]
        image, mask = self.process_path(name)
        return (
            image,
            to_categorical(
                mask,
                self.numOutputClasses
            )
        )

    def __call__(self):
        yield from iter(self)

def dataGeneratorFromIter(**kwargs):
    gen = DataGenIterator(**kwargs)
    imageSig = tf.TensorSpec(shape=gen.imageShape)
    maskSig = tf.TensorSpec(shape=gen.maskShape)
    try:
        dataset = tf.data.Dataset.from_generator(gen, output_signature=(imageSig, maskSig))
    except TypeError:
        # Older version of tensorflow doesn't support output_signature
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(imageSig.dtype, maskSig.dtype),
            output_shapes=(imageSig.shape, maskSig.shape)
        )
    return dataset
