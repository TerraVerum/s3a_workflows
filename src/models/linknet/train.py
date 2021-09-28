from __future__ import annotations

import datetime
import os

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tqdm import tqdm

from features.imagefeats import CompImgsExportWorkflow
from s3a import generalutils as gutils
from src.utils import WorkflowDir, RegisteredPath
from utilitys import fns
from .arch import LinkNet
from ..common import DataGenerator, export_training_data

def tversky_index(Y_true, Y_predicted, alpha = 0.7):
    Y_true = K.cast(Y_true, K.floatx())
    Y_true = K.flatten(Y_true)
    Y_predicted = K.flatten(Y_predicted)
    true_positive = K.sum(Y_true * Y_predicted)
    false_negative = K.sum(Y_true * (1 - Y_predicted))
    false_positive = K.sum((1 - Y_true) * Y_predicted)
    ti = (true_positive + 1.0) / (true_positive + alpha * false_negative + (1 - alpha) * false_positive)
    return ti

def focal_tversky_loss(Y_true, Y_predicted, gamma = 0.75):
    ti = tversky_index(Y_true, Y_predicted)
    ftl = K.pow((1 - ti), gamma)
    return ftl

def dice_coefficient(Y_true, Y_predicted, smoothness=1.0):
    Y_true = K.cast(Y_true, K.floatx())
    Y_true = K.flatten(Y_true)
    Y_predicted = K.flatten(Y_predicted)
    Y_predicted = K.cast(Y_predicted, K.floatx())
    intersection = K.sum(Y_true * Y_predicted)
    dc = (2.0 * intersection + smoothness) / (K.sum(Y_true) + K.sum(Y_predicted) + smoothness)
    return dc

class LinkNetWorkflow(WorkflowDir):
    # Generated during workflow
    graphs_dir = RegisteredPath()
    saved_training_weights_dir = RegisteredPath()
    saved_models_dir = RegisteredPath()
    checkpoints_dir = RegisteredPath()
    predictions_dir = RegisteredPath()

    def run(self, img_export_wf: CompImgsExportWorkflow, number_to_label_map: pd.Series=None):
        base_path = self.workflow_dir
        # devices = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
        devices = ["/gpu:0"]
        strategy = tf.distribute.MirroredStrategy(devices)

        train_ids, validation_ids, test_ids = [
            fns.naturalSorted((img_export_wf.label_masks_dir / typ).glob('*.png'))
            for typ in img_export_wf.ALL_DATA_TYPE_NAMES
        ]

        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0000, patience=10)

        loss = focal_tversky_loss
        optimizer = Adam(learning_rate=0.0005)
        batch_size = 1
        # batch_size = 16
        # batch_size = 32
        epochs = 1

        image_size = 512
        height = image_size
        width = image_size
        color_map = "viridis"
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        training_name = date_time

        generators = [
            DataGenerator(
                id_list,
                img_export_wf.images_dir,
                img_export_wf.label_masks_dir,
                batch_size,
                (height, width),
                number_to_label_map,
                shuffle=True,
            )
            for id_list in (train_ids, validation_ids, test_ids)
        ]
        train_generator, val_generator, test_generator = generators
        train_steps, val_steps, test_steps = [len(g) for g in generators]

        # Same for all generators
        num_output_classes = train_generator.num_output_classes
        with (strategy.scope()):
            mean_iou = MeanIoU(num_classes=num_output_classes)
            metrics = ["accuracy", mean_iou, dice_coefficient]
            linknet_model = LinkNet(height, width, num_output_classes).get_model()
            linknet_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        linknet_training_path = [linknet_model.get_weights()]
        linknet_tensorboard = TensorBoard(
            log_dir=self.graphs_dir / training_name,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        linknet_modelcheckpoint = ModelCheckpoint(
            filepath=self.checkpoints_dir / training_name / "{epoch:02d}.h5",
            verbose=0,
            save_weights_only=True,
            save_freq="epoch",
        )
        linknet_weight_saving = LambdaCallback(
            on_epoch_end=(lambda epoch, logs: linknet_training_path.append(linknet_model.get_weights()))
        )
        linknet_callbacks = [
            linknet_tensorboard,
            linknet_modelcheckpoint,
            linknet_weight_saving,
            early_stopping,
        ]

        linknet_model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=linknet_callbacks,
        )

        linknet_model.evaluate(test_generator, steps=test_steps)

        np.save(
            self.saved_training_weights_dir / f"{training_name}.npy",
            np.array(linknet_training_path, dtype=object),
            )
        linknet_model.save(self.saved_models_dir / f"{training_name}.h5")

        self.save_predictions(linknet_model, training_name)
        export_training_data(base_path / "Graphs", training_name)

    def save_predictions(self, model, training_name, num_classes=None, color_map='viridis'):
        """
        Generates the prediction masks associated with a specific model on entire Test set of the dataset. Saves the files in Binary, Rescaled, and Rescaled RGB versions.
        :param model: The Neural Network model file to generate the predictions of the data.
        :param training_name: The string of the unique training session name the model is associated with.
        :param num_classes: Total number of classes in all train/test/validate data
        :param color_map: A string of the Matplotlib color map to use for the generated RGB ground truth segmentation masks. Acceptable color maps are restrained to the following: https://matplotlib.org/stable/tutorials/colors/colormaps.html.
        """
        wf = CompImgsExportWorkflow(self.workflow_dir)
        image_path = wf.images_dir / "test"
        prediction_path = self.predictions_dir/self.name/training_name
        prediction_wf = CompImgsExportWorkflow(self.predictions_dir)
        test_images = os.listdir(image_path)
        for ti in tqdm(test_images, desc=f"Saving Predictions to {prediction_path}"):
            img = gutils.cvImread_rgb(image_path / ti, cv.IMREAD_UNCHANGED)
            img = np.array([img], dtype=np.uint8)
            prediction = model.predict(img)[0]
            prediction = np.argmax(prediction, axis=-1).astype(np.uint8)
            prediction_wf.generate_colored_mask(prediction, prediction_wf.label_masks_dir, num_classes)
            prediction_wf.generate_colored_mask(prediction, prediction_wf.rgb_masks_dir, num_classes, color_map)