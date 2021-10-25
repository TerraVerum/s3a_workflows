from __future__ import annotations

import gzip
import pickle

import cv2 as cv
import numpy as np
import pandas as pd
from s3a.generalutils import cvImread_rgb
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from .utils import WorkflowDir, RegisteredPath, NestedWorkflow
from .tvtsplit import TrainValidateTestSplitWorkflow


class FeatureTransformerWorkflow(WorkflowDir):
    transformersPath = RegisteredPath()
    ldaAccuracyFile = RegisteredPath('.npy')

    # imageFeaturesDir = RegisteredPath()
    def getFeatsLabels(self, df: pd.DataFrame, labels):
        """
        Turns a dataframe with X rows of MxNx3 component images into a (X x M*N*3) 2 dimensional array where
        each pixel is a feature and each row is a new sample
        """
        feats = np.vstack(df['image'].apply(np.ndarray.ravel))
        inverse = pd.Series(index=labels.values, data=labels.index)
        labels = inverse[df['label'].to_numpy()].values
        return feats, labels

    # @fns.dynamicDocstring(transformers=list(transformers))
    def runWorkflow(
        self,
        parent: NestedWorkflow,
        featureImageShape=(50,50),
        grayscale=False,
        partialFitSize=1000
    ):
        """
        Fits feature transformers like LDA, PCA, etc. on pixel feature data
        :param parent: Parent NestedWorkflow containing ComponentImagesWorkflow
        :param featureImageShape: Images are resized to this shape before being processed through transforers.
          The number of channels is preserved.
       :param grayscale: If *True*, images are first converted to grayscale.
       :param partialFitSize: With lots of images, they often do not all fit in memory at the same time.
         This determines the number if images in a batch during a call to ``transformer.partial_fit
        """
        tvt = parent.get(TrainValidateTestSplitWorkflow)
        summaryDf = pd.read_csv(tvt.filteredSummaryFile, index_col='dataType')

        def batchMaker(dataType):
            if dataType == 'train':
                base = tvt.trainDir
            else:
                base = tvt.testDir
            return BatchGenerator(
                [base / 'images' / file for file in summaryDf.loc[dataType, 'compImageFile']],
                1000,
                (50, 50)
            )

        gen = batchMaker('train')
        pca = self.trainPca(gen)
        lda, numFeatures = self.trainLda(
            pca,
            batchMaker('train'),
            summaryDf.loc['train', 'numericLabel'].to_numpy()
        )
        for obj in lda, pca:
            with gzip.open(self.transformersPath/f'{type(obj).__name__}.pkl', 'wb') as ofile:
                pickle.dump(obj, ofile)

        testGen = batchMaker('test')
        outs = []
        for batch in testGen:
            outs.append(lda.predict(pca.transform(batch)[:, :numFeatures]))
        outs = np.concatenate(outs)
        accuracy = np.count_nonzero(outs == summaryDf.loc['test', 'numericLabel']) / outs.size
        print(f'accuracy: {round(accuracy, 3)}')
        np.save(self.ldaAccuracyFile, outs)

    def trainPca(self, imageGen):
        pca = IncrementalPCA()
        for batch in tqdm(imageGen, 'Training PCA', total=len(imageGen)):
            pca.partial_fit(batch)
        return pca

    def trainLda(self, pca: IncrementalPCA, imageGen, labels, keepVariance=0.9, returnNumFeatures=True):
        lda = LinearDiscriminantAnalysis()
        numFeatures = np.argmin(np.cumsum(pca.explained_variance_ratio_) < keepVariance) + 1
        X_vec = np.empty((len(labels), numFeatures))
        offset = 0
        for batch in tqdm(imageGen, 'Transforming for lda'):
            X_vec[offset:offset + batch.shape[0], :] = pca.transform(batch.reshape(batch.shape[0], -1))[:, :numFeatures]
            offset = offset + batch.shape[0]
        print('Fitting lda...')
        lda.fit(X_vec, labels)
        if returnNumFeatures:
            return lda, numFeatures
        return lda

class BatchGenerator:
    def __init__(self, allImages, batchSize, outputShape, grayscale=False):
        sampleImage = cv.imread(str(allImages[0]), cv.IMREAD_UNCHANGED)
        numChannels = sampleImage.shape[2] if sampleImage.ndim > 2 else 1
        raveledSize = np.prod((*outputShape, numChannels))
        batchSize = int(max(raveledSize, batchSize))
        batchSize = min(len(allImages), batchSize)
        # Make sure batch divides into images
        allImages = allImages[:(len(allImages)//batchSize)*batchSize]
        self.allImages = allImages
        self.batchSize = batchSize
        self.grayscale = grayscale
        self.outputShape = outputShape
        self.raveledSize = raveledSize

    def __iter__(self):
        output = np.empty((self.batchSize, self.raveledSize), dtype='float32')
        counter = 0
        for image in self.allImages:
            im = cvImread_rgb(image, cv.IMREAD_UNCHANGED)
            if self.grayscale and im.ndim > 2:
                im = im.mean(2).astype(im.dtype)
            output[counter, :] = cv.resize(im[...,:3], self.outputShape[::-1]).ravel()
            counter += 1
            if counter >= self.batchSize:
                yield output
                counter = 0

    def __len__(self):
        return len(self.allImages)//self.batchSize
