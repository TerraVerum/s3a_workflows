from __future__ import annotations

import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm

from utilitys import fns

from .compimgs import ComponentImagesWorkflow
from .utils import WorkflowDir, RegisteredPath, NestedWorkflow

class ImageToFeatureWorkflow(WorkflowDir):
    transformersPath = RegisteredPath()
    # imageFeaturesDir = RegisteredPath()

    def getFeatsLabels(self, df: pd.DataFrame, labels, returnDf=False):
        """
        Turns a dataframe with X rows of MxNx3 component images into a (X x M*N*3) 2 dimensional array where
        each pixel is a feature and each row is a new sample
        """
        feats = np.vstack(df['image'].apply(np.ndarray.ravel))
        inverse = pd.Series(index=labels.values, data=labels.index)
        labels = inverse[df['label'].to_numpy()].values
        if returnDf:
            return feats, labels, df
        return feats, labels

    def fitSaveTransformer(self, transformer, featsLabelsOrDf: pd.DataFrame | tuple, labels):
        if not isinstance(featsLabelsOrDf, tuple):
            featsLabels = self.getFeatsLabels(featsLabelsOrDf, labels)
        else:
            featsLabels = featsLabelsOrDf
        transformer.fit(*featsLabels)
        saveName = self.transformersPath / (type(transformer).__name__ + '.pkl')
        with open(saveName, 'wb') as ofile:
            pickle.dump(transformer, ofile)
        return transformer

    def runWorkflow(self, parent: NestedWorkflow):
        # Images are too big natively to fit into one dataframe, so resize them
        compImgsWf = parent.get(ComponentImagesWorkflow)
        def readFunc(file: Path):
            df_ = pd.read_pickle(file)
            for idx, img in df_['image'].iteritems():
                df_.at[idx, 'image'] = cv.resize(img, (50,50))
            return df_
        df = fns.readDataFrameFiles(compImgsWf.compImgsDir, readFunc)
        # Ensure classes aren't overrepresented and blanks aren't an actual class
        df: pd.DataFrame = df[df['label'].str.len() > 0]
        df = df.groupby('label').apply(lambda el: el.sample(n=min(len(el), 150), random_state=10)).reset_index(drop=True)
        featsLabels = self.getFeatsLabels(df, compImgsWf.createGetLabelMapping())
        for xformer in tqdm([PCA(), LDA()], desc='Fitting transformers'):
            self.fitSaveTransformer(xformer, featsLabels, compImgsWf.createGetLabelMapping())


