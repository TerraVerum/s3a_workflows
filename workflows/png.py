import os
from pathlib import Path

import numpy as np
import pandas as pd
from s3a import generalutils as gutils
from utilitys import fns

from .compimgs import ComponentImagesWorkflow
from .constants import DEBUG
from .utils import WorkflowDir, RegisteredPath, NestedWorkflow

class PngExportWorkflow(WorkflowDir):

    TRAIN_NAME = 'train'
    VALIDATION_NAME = 'validation'
    TEST_NAME = 'test'

    ALL_DATA_TYPE_NAMES = [TRAIN_NAME, VALIDATION_NAME, TEST_NAME]

    imagesDir = RegisteredPath()

    labelMasksDir = RegisteredPath()

    summariesDir = RegisteredPath()
    summaryFile = RegisteredPath('.csv')

    def runWorkflow(self, parent: NestedWorkflow):
        """
        Automatically generates the Neural Network data in an appropriate directory structure
        and format in the base path with the resized and padded images and corresponding binary Masks.
        """
        compImgsWf = parent.get(ComponentImagesWorkflow)
        files = np.array(list(compImgsWf.compImgsDir.glob('*.*')))
        stems = [f.stem for f in files]
        if self.summaryFile.exists():
            summary = pd.read_csv(self.summaryFile)
            newFiles = fns.naturalSorted(files[np.isin(stems, [Path(f).stem for f in summary['imageFile']], invert=True)])
        else:
            newFiles = fns.naturalSorted(files)

        fns.mproc_apply(
            self._exportSinglePcbImage,
            newFiles,
            descr="Exporting Png Files",
            showProgress=True,
            applyAsync=True,
            debug=DEBUG,
        )

        self.createMergedSummaries()

    def _exportSinglePcbImage(self, compImgsFile):
        outDf = pd.read_pickle(compImgsFile)
        outDf['imageFile'] = compImgsFile.with_suffix('.png').name

        exportedImgs = []
        for index, row in gutils.pd_iterdict(outDf, index=True):
            imageName = self._exportSingleComp(index, row)
            exportedImgs.append(imageName)
        outDf['compImageFile'] = exportedImgs
        outDf: pd.DataFrame
        outDf.drop(columns=['labelMask', 'image']).to_csv(
            self.summariesDir / compImgsFile.with_suffix('.csv').name,
            index=False
        )

    def _exportSingleComp(self, index, row: dict):
        exportName = f'{os.path.splitext(row["imageFile"])[0]}_id_{index}.png'

        for retKey, dir_ in zip(
            ['image', 'labelMask'],
            [self.imagesDir, self.labelMasksDir]
        ):
            saveName = dir_/exportName
            gutils.cvImsave_rgb(saveName, row[retKey])

        return exportName

    def createMergedSummaries(self):
        concatDf = fns.readDataFrameFiles(self.summariesDir, pd.read_csv)
        concatDf.to_csv(self.summaryFile, index=False)
        return concatDf
