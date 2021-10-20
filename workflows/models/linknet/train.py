from __future__ import annotations

import os.path
import typing as t
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.models import load_model
from s3a import generalutils as gutils
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tqdm import tqdm
from utilitys import fns
from utilitys.typeoverloads import FilePath

from .arch import LinkNet
from ..common import DataGenerator, export_training_data
from ...png import PngExportWorkflow
from ...tvtsplit import LabelMaskResolverWorkflow, TrainValidateTestSplitWorkflow
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
    savedModelFile = RegisteredPath('.h5')
    checkpointsDir = RegisteredPath()
    predictionsDir = RegisteredPath()

    def runWorkflow(
        self,
        parent: NestedWorkflow,
        learningRate=0.001,
        batchSize=8,
        epochs=1000,
        numPredictionsDuringTrain=0
    ):
        """
        Trains a LinkNet model
        :param parent: NestedWorkflow with TrainValidateTestSplitWorkflow
        :param learningRate: Adam learning rate during training
        :param batchSize: train batch size
        :param epochs: Number of epochs to train. Early stopping is implemented, so not all epochs might be reached
        :param numPredictionsDuringTrain: Nothing happens if <= 0. Otherwise, at the end of each epoch, prediction
          images will be generated on this many samples of the holdout set for visualization purposes
        """
        # Find out how many digits are needed to store the epoch number
        epochFormatter = len(str(epochs))
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

        linknetTensorboard = TensorBoard(
            log_dir=self.graphsDir,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        modelCPFormat = f'{{epoch:0{epochFormatter}d}}'
        linknetModelcheckpoint = ModelCheckpoint(
            filepath=self.checkpointsDir / (modelCPFormat + '.h5'),
            verbose=0,
            save_weights_only=True,
            save_freq="epoch",
        )
        linknetCallbacks = [
            linknetTensorboard,
            linknetModelcheckpoint,
            earlyStopping,
        ]

        if numPredictionsDuringTrain > 0:
            rng = np.random.default_rng(42)
            testFiles = rng.choice(
                fns.naturalSorted(tvtWf.testDir.joinpath(PEW.imagesDir).glob('*.png')),
                numPredictionsDuringTrain,
                replace=False
            )
            def predictAfterEpoch(epoch, logs):
                outDir = self.predictionsDir/modelCPFormat.format(epoch=epoch)
                self.savePredictions(linknetModel, testFiles, outDir)
                labels = pd.read_csv(tvtWf.classInfoFile, index_col='numeric_class')['label']
                PEW(outDir).createOverlays(labels=labels)
            linknetCallbacks.append(LambdaCallback(on_epoch_end=predictAfterEpoch))

        linknetModel.fit(
            trainGenerator,
            steps_per_epoch=trainSteps,
            epochs=epochs,
            validation_data=valGenerator,
            validation_steps=valSteps,
            callbacks=linknetCallbacks,
        )

        linknetModel.evaluate(testGenerator, steps=testSteps)

        linknetModel.save(self.savedModelFile)
        # Only need to save final if there was no intermediate saving
        if numPredictionsDuringTrain <= 0:
            testFiles = fns.naturalSorted((tvtWf.testDir/PEW.imagesDir).glob('*.png'))
            self.savePredictions(linknetModel, testFiles)
        export_training_data(self.graphsDir, self.name)

    def savePredictions(self, model, testImagePaths, outputDir=None):
        """
        Generates the prediction masks associated with a specific model on entire Test set of the dataset. Saves the files in Binary, Rescaled, and Rescaled RGB versions.
        :param model: The Neural Network model file to generate the predictions of the data.
        :param testImagePaths: Images to save the predictions of
        :param outputDir: Where to save the output predictions. Defaults to `self.predictionsDir` if unspecified
        """
        if outputDir is None:
            outputDir = self.predictionsDir
        outputDir = Path(outputDir)
        outputDir.mkdir(exist_ok=True)
        legendVisible = None
        if 'parent' in self.input:
            tvtWf = self.input['parent'].get(TrainValidateTestSplitWorkflow)
            pngWf = self.input['parent'].get(PngExportWorkflow)
            labelMap = pd.read_csv(tvtWf.classInfoFile, index_col='numeric_class')['label']
            if len(labelMap.unique()) <= 2:
                # No need for a legend if there's only foreground/background
                legendVisible = pngWf.compositor.legend.isVisible()
                pngWf.compositor.legend.setVisible(False)
        else:
            pngWf = PngExportWorkflow('')
            labelMap = None
        compositorProps = pngWf.compositor.propertiesProc
        oldSettings = dict(compositorProps.input)
        compositorProps.run(opacity=0.7, colormap='viridis')
        for file in tqdm(testImagePaths, desc=f"Saving Predictions to {outputDir}"):
            img = gutils.cvImread_rgb(file, cv.IMREAD_UNCHANGED)
            img = np.array([img], dtype=np.uint8)
            prediction = model.predict(img)[0]
            prediction = np.argmax(prediction, axis=-1).astype(np.uint8)
            outFile = outputDir.joinpath(os.path.basename(file)).with_suffix('.jpg')
            pngWf.overlayMaskOnImage(img[0], prediction, labelMap, outFile)
        compositorProps.run(**oldSettings)
        if legendVisible is not None:
            pngWf.compositor.legend.setVisible(legendVisible)

    def loadAndTestModel(
        self,
        testImagePaths: t.Sequence[FilePath]=None,
        modelFile: FilePath=None,
        outputDir=None
    ):
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
            self.savePredictions(model, testImagePaths, outputDir)
        return model

    def loadAndTestWeights(self, testImagePaths: t.Sequence[FilePath]=None, numClasses=None, weightsFile=None):
        if numClasses is None and 'parent' in self.input:
            numClasses = len(pd.read_csv(self.input['parent'].get(TrainValidateTestSplitWorkflow).classInfoFile, usecols=['label']))
        model = LinkNet(512, 512, numClasses).get_model()
        model.load_weights(weightsFile)
        if testImagePaths:
            self.savePredictions(model, testImagePaths)
        return model
