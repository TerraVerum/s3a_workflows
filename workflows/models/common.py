from __future__ import annotations

import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils.np_utils import to_categorical

from s3a import generalutils as gutils

from workflows.imagefeats import TrainValTestWorkflow as TVTW

class DataGenerator(Sequence):
    """
    A class that serves as a custom data generator for the Neural Network pipeline.
    Inputs different file names, reads in image files and masks in batches based on batch size,
     and returns the batch of data for a specific data type of Train, Validation, or Test.
    """

    def __init__(
        self,
        owned_image_names: list[Path | str],
        images_dir: Path,
        labelMasksDir,
        batch_size,
        image_shape,
        numOutputClasses,
        shuffle
    ):
        """
        :param owned_images: list
            A list of Path object file directories for the specific data type.
        :param images_dir: Path
            Images to use as inputs data
        :param labelMasksDir: Path
            Masks to use as ground truth outputs
        :param batch_size: int
            A int of the batch size of the training of the Neural Network.
        :param image_shape: tuple
            A size two tuple of the height and width of the image.
        :param numOutputClasses: Number of total classes present
        :param shuffle: bool
            A boolean that indicates whether the data should be shuffled at the end of each epoch.
        """
        np.random.seed(22)
        names = []
        for fpath in owned_image_names:
            if isinstance(fpath, Path):
                fpath = fpath.name
            names.append(fpath)
        self.owned_image_names = names
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.owned_image_names))
        self.numOutputClasses = numOutputClasses
        self.labelMasksDir = labelMasksDir
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches in an epoch of the data.
        """
        return int(np.floor(len(self.owned_image_names) / self.batch_size))

    def __getitem__(self, index):
        """
        Gets the batch of data with the specified batch size at a specific index of the data.
        """
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indexes = self.indexes[start:end]
        image_names_temp = [self.owned_image_names[i] for i in indexes]
        X, Y = self.__data_generation(image_names_temp)
        return X, Y

    def on_epoch_end(self):
        """
        Gets executed at the end of each epoch and recollects all file ids and shuffles the ids if the shuffle parameter is set.
        """
        self.indexes = np.arange(len(self.owned_image_names))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_names):
        """
        Returns an array of the images and segmentation masks for a batch of data.
        :param image_names: The file ids of the specific batch of data.
        """
        num_classes = self.numOutputClasses
        images = np.empty((len(image_names), *self.image_shape, 3), dtype=np.uint8)
        masks = np.empty((len(image_names), *self.image_shape, num_classes), dtype=np.uint8)
        for index, img_file in enumerate(image_names):
            img = gutils.cvImread_rgb(self.images_dir / img_file, cv.IMREAD_UNCHANGED)
            mask = gutils.cvImread_rgb(self.labelMasksDir / img_file, cv.IMREAD_UNCHANGED)
            # Uncomment below to turn to single class
            # mask[mask > 0] = 1
            images[index, ...] = img
            mask = to_categorical(mask, num_classes=num_classes, dtype=np.uint8)
            masks[index, ...] = mask
        return images, masks

RNG_SEED = 42

def export_training_data(graph_path, training_name):
    """
    Uses the tensorboard library to generate a CSV file of the training data for a specific model, with the data containing the specific Loss, Accuracy, Mean IoU, and Dice Coefficient values at each epoch the model is trained. Saves the csv files in the <BASE_PATH>/Graphs/<NETWORK_MODEL>/Training Values path with the training session name as the file name.
    :param graph_path: The pathlib object of the Graph directory of the specific Neural Network, for example for the LinkNet mdoel this would <BASE_PATH>/Graphs/LinkNet
    :param training_name: The string of the unique training session name the model is associated with.
    :return values_df: The dataframe of the different metric values for each training epoch.
    """
    csv_path = f'{graph_path}.csv'
    if os.path.isfile(csv_path):
        values_df = pd.read_csv(csv_path)
    else:
        ea_train = event_accumulator.EventAccumulator(str(graph_path / 'train'))
        ea_validation = event_accumulator.EventAccumulator(str(graph_path / 'validation'))
        ea_train.Reload()
        ea_validation.Reload()
        train_values = []
        validation_values = []
        scalars = ea_train.Tags()["scalars"]
        if not scalars:
            return
        for scalar in scalars:
            train_values.append([scalar_event.value for scalar_event in ea_train.Scalars(scalar)])
            validation_values.append([scalar_event.value for scalar_event in ea_validation.Scalars(scalar)])
        values = train_values + validation_values
        values = np.array(values)
        values = np.transpose(values)
        values_df = pd.DataFrame(values)
        values_df.index.name = "Epoch"
        values_df.columns = pd.MultiIndex.from_product(
            [["Train", "Validation"], ["Loss", "Accuracy", "Mean IoU", "Dice Coefficient"]]
        )
        values_df.to_csv(csv_path)
    return values_df
