from __future__ import annotations

import datetime
import os

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from s3a import generalutils as gutils
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tqdm import tqdm

from .arch import LinkNet
from ..common import DataGenerator, export_training_data
from ...features.imagefeats import CompImgsExportWorkflow, TrainValTestWorkflow, LabelMaskResolverWorkflow
from ...utils import WorkflowDir, RegisteredPath, AliasedMaskResolver


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

    default_config = dict(
        learning_rate=0.0005,
        epochs=1000
    )

    def run(self, tvt_wf: TrainValTestWorkflow):
        base_path = self.workflow_dir
        # devices = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
        devices = ["/gpu:0"]
        strategy = tf.distribute.MirroredStrategy(devices)

        summary_df = pd.read_csv(tvt_wf.filtered_summary_file)
        tvt_files = []
        for typ in tvt_wf.TRAIN_NAME, tvt_wf.VALIDATION_NAME, tvt_wf.TEST_NAME:
            tvt_files.append(summary_df.loc[summary_df['dataType'] == typ, 'compImageFile'])

        class_info = pd.read_csv(tvt_wf.class_info_file)

        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0000, patience=10)

        loss = focal_tversky_loss
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        batch_size = 1
        # batch_size = 16
        # batch_size = 32

        image_size = 512
        height = image_size
        width = image_size
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        training_name = date_time

        EW = CompImgsExportWorkflow
        generators = [
            DataGenerator(
                name_list,
                dir_/EW.images_dir,
                dir_/EW.label_masks_dir,
                batch_size,
                (height, width),
                np.max(class_info['numeric_class']),
                shuffle=True,
            )
            for name_list, dir_ in zip(tvt_files, [tvt_wf.train_dir, tvt_wf.val_dir, tvt_wf.test_dir])
        ]
        train_generator, val_generator, test_generator = generators
        train_steps, val_steps, test_steps = [len(g) for g in generators]

        # Same for all generators
        num_output_classes = train_generator.num_output_classes + 1
        with (strategy.scope()):
            mean_iou = MeanIoU(num_classes=num_output_classes)
            metrics = ["accuracy", mean_iou, dice_coefficient]
            linknet_model = LinkNet(height, width, num_output_classes).get_model()
            linknet_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        linknet_training_path = [linknet_model.get_weights()]
        linknet_tensorboard = TensorBoard(
            log_dir=self.graphs_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        linknet_modelcheckpoint = ModelCheckpoint(
            filepath=self.checkpoints_dir / "{epoch:02d}.h5",
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
            epochs=self.config['epochs'],
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

        self.save_predictions(linknet_model, training_name, tvt_wf.test_dir, num_output_classes-1)
        export_training_data(base_path / "Graphs", training_name)

    def save_predictions(self, model, training_name, test_image_paths, num_classes=None):
        """
        Generates the prediction masks associated with a specific model on entire Test set of the dataset. Saves the files in Binary, Rescaled, and Rescaled RGB versions.
        :param model: The Neural Network model file to generate the predictions of the data.
        :param training_name: The string of the unique training session name the model is associated with.
        :param test_image_paths: Images to save the predictions of
        :param num_classes: Total number of classes in all train/test/validate data
        """
        prediction_path = self.predictions_dir/training_name
        prediction_path.mkdir(exist_ok=True)
        mask_wf = LabelMaskResolverWorkflow(self.predictions_dir, create_dirs=True)
        resolver = AliasedMaskResolver(np.arange(num_classes) if num_classes else None)
        for file in tqdm(test_image_paths, desc=f"Saving Predictions to {prediction_path}"):
            img = gutils.cvImread_rgb(file, cv.IMREAD_UNCHANGED)
            img = np.array([img], dtype=np.uint8)
            prediction = model.predict(img)[0]
            prediction = np.argmax(prediction, axis=-1).astype(np.uint8)
            mask_wf.run([prediction], resolver)
