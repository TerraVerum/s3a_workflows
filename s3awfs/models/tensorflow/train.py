from __future__ import annotations

import os.path
import shutil
import typing as t
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from s3a import generalutils as gutils
from tensorboard.backend.event_processing import event_accumulator
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from utilitys import fns, ProcessIO
from utilitys.typeoverloads import FilePath

from s3awfs.png import PngExportWorkflow
from s3awfs.tvtsplit import TrainValidateTestSplitWorkflow
from s3awfs.utils import WorkflowDir, RegisteredPath
from .datagen import (
    dataGeneratorFromIter,
    SequenceDataGenerator,
    SquareMaskSequenceDataGenerator
)

def makeTensorflowStrategy(strategyClass='MirroredStrategy', devices=('/cpu:0',)):
    strat = getattr(tf.distribute, strategyClass)
    return ProcessIO(strategy=strat(devices))

class TensorflowTrainingWorkflow(WorkflowDir):
    # Generated during workflow
    graphsDir = RegisteredPath()
    savedModelFile = RegisteredPath('.h5')
    checkpointsDir = RegisteredPath()
    predictionsDir = RegisteredPath()

    def runWorkflow(
        self,
        model: Model | FilePath=None,
        resizeOpts=None,
        customObjects: dict=None,
        compileOpts: dict=None,
        learningRate=0.001,
        batchSize=12,
        epochs=1000,
        predictionDuringTrainPath=None,
        initialEpoch=0,
        workers=1,
        bufferSize=1000,
        tensorboardUpdatesPerEpoch=5,
        strategy=None,
        convertMasksToBbox=False
    ):
        """
        Trains a LinkNet model
        :param model: Keras model to be trained. Can either be a h5 model file or the Model object itself.
          defaults to ``self.savedModelFile` if unspecified
        :param resizeOpts: Options for how images are resized, must contain a 'shape' :key
          with the shape of the model input
        :param customObjects: Custom objects to be passed during model loading, if any
        :param compileOpts: Unpacked into `model.compile()` along with Adam loss
        :param learningRate: Adam learning rate during training
        :param batchSize: train batch size
        :param epochs: Number of epochs to train. Early stopping is implemented, so not all epochs might be reached
        :param predictionDuringTrainPath: At the end of each epoch, predictions will be run on images from this
          directory if it is specified. They will be visible in a subfolder of ``predictionsDir`` based on their
          epoch number
        :param initialEpoch: If above 0, model weights from this epoch will be loaded and the epoch counter will
          resume at startEpoch+1. Should match the integer representation of the checkpoint model name
        :param workers: Number of CPU workers for data generation during training. If greater than 1, this uses
          multiprocessing with ``workers`` number of cores.
        :param bufferSize: How large of a shuffle buffer to create. Prefetch buffer is 1/10 of this size
          Only used if workers > 1
        :param tensorboardUpdatesPerEpoch: Number of times per training epoch tensorboard should update
        :param strategy: Tensorflow strategy to use during model compilation/training
        :param convertMasksToBbox: If *True*, semantic masks will be converted into bboxes before training.
          This is only useful to experiment with the effect of using bboxes on training accuracy
        """
        # Find out how many digits are needed to store the epoch number
        numEpochDigits = len(str(epochs))
        # Give a formatter that takes into account the starting epoch to avoid overwrites
        epochFormatter = f'{{epoch:0{numEpochDigits}d}}'
        # devices = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]

        # Graphs don't play nice with old files
        # Directory watchers are spawned early, so delete these directories before they have a chance to see them
        for item in self.graphsDir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Bbox masks require 1 worker
        if convertMasksToBbox:
            workers = 1

        strategy = strategy or tf.distribute.get_strategy()
        tvtWf = self.parent.get(TrainValidateTestSplitWorkflow)
        summaryDf = pd.read_csv(tvtWf.filteredSummaryFile)
        tvtFiles = []
        for typ in tvtWf.TRAIN_NAME, tvtWf.VALIDATION_NAME, tvtWf.TEST_NAME:
            tvtFiles.append(summaryDf.loc[summaryDf['dataType'] == typ, 'compImageFile'])

        classInfo = pd.read_csv(tvtWf.classInfoFile)

        earlyStopping = EarlyStopping(monitor="val_loss", min_delta=0.0000, patience=10)

        PEW = PngExportWorkflow
        tvtWf.resolver.setClassInfo(classInfo)

        batchKwargs = {}
        if workers > 1 and tf.__version__ > '2.5':
            batchKwargs['num_parallel_calls'] = workers

        if workers <= 1:
            constructor = SquareMaskSequenceDataGenerator if convertMasksToBbox else SequenceDataGenerator
        else:
            def constructor(**kwargs):
                return dataGeneratorFromIter(**kwargs) \
                    .shuffle(bufferSize, reshuffle_each_iteration=True) \
                    .batch(batchSize, **batchKwargs) \
                    .prefetch(max(1, bufferSize // 10))
        resizeOpts = resizeOpts or dict(shape=(512,512))
        generators = [
            constructor(
                ownedImageNames=nameList,
                imagesDir=dir_/PEW.imagesDir,
                labelMasksDir=dir_/PEW.labelMasksDir,
                imageShape=(*resizeOpts['shape'], 3),
                numOutputClasses=tvtWf.resolver.numClasses,
                batchSize=batchSize,
                shuffle=True
            )
            for nameList, dir_ in zip(tvtFiles, [tvtWf.trainDir, tvtWf.validateDir, tvtWf.testDir])
        ]
        trainGenerator, valGenerator, testGenerator = generators

        def calcNumBatches(fileList):
            return int(np.floor(len(fileList) / batchSize))

        trainSteps, valSteps, testSteps = [calcNumBatches(lst) for lst in tvtFiles]

        overwriteFile = None
        if model is None:
            model = self.savedModelFile
        compileOpts = compileOpts or {}
        compileOpts['optimizer'] = Adam(learning_rate=learningRate)
        with strategy.scope():
            if isinstance(model, FilePath.__args__):
                overwriteFile = model
                model = load_model(model, custom_objects=customObjects)
            model.compile(**compileOpts)


        if tensorboardUpdatesPerEpoch <= 1:
            updateFreq = 'epoch'
        else:
            updateFreq = calcNumBatches(tvtFiles[0])//tensorboardUpdatesPerEpoch
        linknetTensorboard = TensorBoard(
            log_dir=self.graphsDir,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq=updateFreq
        )
        # Allow save directory to have an epoch offset
        linknetModelcheckpoint = ModelCheckpoint(
            filepath=self.checkpointsDir / (epochFormatter + '.h5'),
            verbose=0,
            save_freq="epoch",
        )
        linknetCallbacks = [
            linknetModelcheckpoint,
            earlyStopping,
        ]
        if tensorboardUpdatesPerEpoch > 0:
            linknetCallbacks.append(linknetTensorboard)

        if predictionDuringTrainPath is not None:
            predictionDuringTrainPath = Path(predictionDuringTrainPath)
            labels = pd.read_csv(tvtWf.classInfoFile, index_col='numeric_class', na_filter=False)['label']
            def predictAfterEpoch(epoch, logs):
                # Calculate test files in function so they can be swapped out by the user in between
                # epochs if desired
                testFiles = fns.naturalSorted(predictionDuringTrainPath.glob('*.png'))

                # Add 1 to match naming scheme of ModelCheckpoint
                epoch += 1
                outDir = self.predictionsDir/epochFormatter.format(epoch=epoch)
                self.savePredictions(model, testFiles, outDir)
                PEW(outDir).createOverlays(labels=labels)
            linknetCallbacks.append(LambdaCallback(on_epoch_end=predictAfterEpoch))

        if workers > 1:
            moreKwargs = dict(use_multiprocessing=True, workers=workers)
        else:
            moreKwargs = {}
        model.fit(
            trainGenerator,
            steps_per_epoch=trainSteps,
            epochs=epochs,
            validation_data=valGenerator,
            validation_steps=valSteps,
            callbacks=linknetCallbacks,
            initial_epoch=initialEpoch,
            **moreKwargs
        )

        model.evaluate(testGenerator, steps=testSteps)

        model.save(self.savedModelFile)
        # Only need to save final if there was no intermediate saving
        if predictionDuringTrainPath is None:
            testFiles = fns.naturalSorted(tvtWf.testDir.joinpath(PEW.imagesDir).glob('*.png'))
            self.savePredictions(model, testFiles)
        self.export_training_data()
        if overwriteFile is not None:
            # Make sure the updated model replaces the source file
            model.save(overwriteFile)
        return ProcessIO(model=model)

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
        try:
            tvtWf = self.parent.get(TrainValidateTestSplitWorkflow)
            pngWf = self.parent.get(PngExportWorkflow)
            labelMap = pd.read_csv(tvtWf.classInfoFile, index_col='numeric_class', dtype=str)['label']
        except (FileNotFoundError, KeyError, AttributeError):
            # Labelmap doesn't exist / parent not specified
            pngWf = PngExportWorkflow('')
            labelMap = None
        if model.output_shape[-1] <= 2:
            # No need for a legend if there's only foreground/background (2 output classes)
            legendVisible = pngWf.compositor.legend.isVisible()
            pngWf.compositor.legend.setVisible(False)
        compositorProps = pngWf.compositor.propertiesProc
        oldSettings = dict(compositorProps.input)
        compositorProps.run(opacity=0.7)
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

    def export_training_data(self):
        """
        Uses the tensorboard library to generate a CSV file of the training data for a specific model, with the data containing the specific Loss, Accuracy, Mean IoU, and Dice Coefficient values at each epoch the model is trained. Saves the csv files in the <BASE_PATH>/Graphs/<NETWORK_MODEL>/Training Values path with the training session name as the file name.
        :return values_df: The dataframe of the different metric values for each training epoch.
        """
        for subpath in 'train', 'validation':
            self.graphsDir.joinpath(subpath).mkdir(exist_ok=True)
        graph_path = self.graphsDir
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
