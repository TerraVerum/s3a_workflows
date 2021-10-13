from __future__ import annotations

import typing as t

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.models import load_model
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from s3a import generalutils as gutils
from tqdm import tqdm
from utilitys import fns
from utilitys.typeoverloads import FilePath

from ...tvtsplit import LabelMaskResolverWorkflow, TrainValidateTestSplitWorkflow
from ...png import PngExportWorkflow
from .arch import LinkNet
from ..common import DataGenerator, export_training_data
from ...utils import WorkflowDir, RegisteredPath, AliasedMaskResolver, NestedWorkflow


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

class LinkNetTrainingWorkflow(WorkflowDir):
    # Generated during workflow
    graphsDir = RegisteredPath()
    savedTrainingWeightsFile = RegisteredPath('.npy')
    savedModelFile = RegisteredPath('.h5')
    checkpointsDir = RegisteredPath()
    predictionsDir = RegisteredPath()

    def runWorkflow(
        self,
        parent: NestedWorkflow,
        learningRate=0.001,
        batchSize=8,
        epochs=1000
    ):
        # devices = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
        devices = ["/gpu:0"]
        strategy = tf.distribute.MirroredStrategy(devices)
        tvtWf = parent.get(TrainValidateTestSplitWorkflow)
        summaryDf = pd.read_csv(tvtWf.filteredSummaryFile)
        tvtFiles = []
        for typ in tvtWf.TRAIN_NAME, tvtWf.VALIDATION_NAME, tvtWf.TEST_NAME:
            tvtFiles.append(summaryDf.loc[summaryDf['dataType'] == typ, 'compImageFile'])

        classInfo = pd.read_csv(tvtWf.classInfoFile)

        earlyStopping = EarlyStopping(monitor="val_loss", min_delta=0.0000, patience=10)

        loss = focal_tversky_loss
        optimizer = Adam(learning_rate=learningRate)

        imageSize = 512
        height = imageSize
        width = imageSize

        PEW = PngExportWorkflow
        tvtWf.resolver.setClassInfo(classInfo)
        generators = [
            DataGenerator(
                nameList,
                dir_/PEW.imagesDir,
                dir_/PEW.labelMasksDir,
                batchSize,
                (height, width),
                tvtWf.resolver.numClasses,
                shuffle=True,
            )
            for nameList, dir_ in zip(tvtFiles, [tvtWf.trainDir, tvtWf.validateDir, tvtWf.testDir])
        ]
        trainGenerator, valGenerator, testGenerator = generators
        trainSteps, valSteps, testSteps = [len(g) for g in generators]

        # Same for all generators
        numOutputClasses = trainGenerator.numOutputClasses
        with (strategy.scope()):
            meanIou = MeanIoU(num_classes=numOutputClasses)
            metrics = ["accuracy", meanIou, dice_coefficient]
            linknetModel = LinkNet(height, width, numOutputClasses).get_model()
            linknetModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        linknetTrainingPath = [linknetModel.get_weights()]
        linknetTensorboard = TensorBoard(
            log_dir=self.graphsDir,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        linknetModelcheckpoint = ModelCheckpoint(
            filepath=self.checkpointsDir / "{epoch:02d}.h5",
            verbose=0,
            save_weights_only=True,
            save_freq="epoch",
        )
        linknetWeightSaving = LambdaCallback(
            on_epoch_end=(lambda epoch, logs: linknetTrainingPath.append(linknetModel.get_weights()))
        )
        linknetCallbacks = [
            linknetTensorboard,
            linknetModelcheckpoint,
            linknetWeightSaving,
            earlyStopping,
        ]

        linknetModel.fit(
            trainGenerator,
            steps_per_epoch=trainSteps,
            epochs=epochs,
            validation_data=valGenerator,
            validation_steps=valSteps,
            callbacks=linknetCallbacks,
        )

        linknetModel.evaluate(testGenerator, steps=testSteps)

        np.save(
            self.savedTrainingWeightsFile,
            np.array(linknetTrainingPath, dtype=object),
        )
        linknetModel.save(self.savedModelFile)
        testFiles = fns.naturalSorted((tvtWf.testDir/PEW.imagesDir).glob('*.png'))
        self.savePredictions(linknetModel, testFiles, numClasses=numOutputClasses)
        export_training_data(self.graphsDir, self.name)

    def savePredictions(self, model, testImagePaths, numClasses=None):
        """
        Generates the prediction masks associated with a specific model on entire Test set of the dataset. Saves the files in Binary, Rescaled, and Rescaled RGB versions.
        :param model: The Neural Network model file to generate the predictions of the data.
        :param testImagePaths: Images to save the predictions of
        :param numClasses: Total number of classes in all train/test/validate data
        """
        predictionPath = self.predictionsDir
        predictionPath.mkdir(exist_ok=True)
        maskWf = LabelMaskResolverWorkflow(predictionPath, createDirs=True)
        resolver = AliasedMaskResolver(np.arange(numClasses) if numClasses else None)
        for file in tqdm(testImagePaths, desc=f"Saving Predictions to {predictionPath}"):
            img = gutils.cvImread_rgb(file, cv.IMREAD_UNCHANGED)
            img = np.array([img], dtype=np.uint8)
            prediction = model.predict(img)[0]
            prediction = np.argmax(prediction, axis=-1).astype(np.uint8)
            maskWf.runWorkflow([prediction], resolver, [file])

    def loadAndTestModel(self, testImagePaths: t.Sequence[FilePath]=None, numClasses=None, modelFile: FilePath=None):
        if modelFile:
            modelFile = self.workflowDir / modelFile
        else:
            modelFile = self.savedModelFile

        customObjects = {
            'dice_coefficient': dice_coefficient,
            'tversky_index': tversky_index,
            'focal_tversky_loss': focal_tversky_loss,
        }

        model = load_model(modelFile, compile = True, custom_objects = customObjects)
        if testImagePaths:
            self.savePredictions(model, testImagePaths, numClasses)
        return model
