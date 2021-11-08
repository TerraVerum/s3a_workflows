import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from s3a import generalutils as gutils
from utilitys import fns, widgets

from . import constants
from .compimgs import ComponentImagesWorkflow
from .utils import WorkflowDir, RegisteredPath, NestedWorkflow

class PngExportWorkflow(WorkflowDir):

    TRAIN_NAME = 'train'
    VALIDATION_NAME = 'validation'
    TEST_NAME = 'test'

    ALL_DATA_TYPE_NAMES = [TRAIN_NAME, VALIDATION_NAME, TEST_NAME]

    imagesDir = RegisteredPath()
    labelMasksDir = RegisteredPath()
    overlaysDir = RegisteredPath()

    summariesDir = RegisteredPath()
    summaryFile = RegisteredPath('.csv')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compositor = widgets.MaskCompositor()

    def runWorkflow(
        self,
        createOverlays=False,
        overlayColormap=constants.DEFAULT_RGB_CMAP
    ):
        """
        Automatically generates the Neural Network data in an appropriate directory structure
        and format in the base path with the resized and padded images and corresponding binary Masks.
        :param createOverlays: If *True*, ``overlaysDir`` will be populated with every mask superimposed over
          its corresponding image. This can be helpful for visualization, but quite time consuming.
        :param overlayColormap: If ``createOverlays`` is *True*, this is the colormap assigned to mask values
        """
        compImgsWf = self.parent.get(ComponentImagesWorkflow)
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
            debug=constants.DEBUG,
        )

        self.createMergedSummaries()
        if createOverlays:
            self.createOverlays(colormap=overlayColormap)

    def _exportSinglePcbImage(self, compImgsFile):
        outputSummaryName = self.summariesDir / compImgsFile.with_suffix('.csv').name
        if outputSummaryName.exists():
            return

        outDf = ComponentImagesWorkflow.readDataframe(compImgsFile)
        outDf['imageFile'] = compImgsFile.with_suffix('.png').name

        exportedImgs = []
        for index, row in gutils.pd_iterdict(outDf, index=True):
            imageName = self._exportSingleComp(index, row)
            exportedImgs.append(imageName)
        outDf['compImageFile'] = exportedImgs
        outDf: pd.DataFrame
        outDf.drop(columns=['labelMask', 'image']).to_csv(
            outputSummaryName,
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

    def createOverlays(self, labels: pd.Series=None, colormap=constants.DEFAULT_RGB_CMAP):
        """
        Overlays masks on top of images and saves to a new directory
        :param labels: Mapping of numeric mask value to its string label
        :param colormap: Colormap used for mask values
        """
        if labels is None:
            summaries = pd.read_csv(self.summaryFile, dtype=str, na_filter=False, index_col=['numericLabel'])
            labels = summaries['label'].drop_duplicates()
            labels[labels.str.len() == 0] = '<blank>'
        oldProps = dict(self.compositor.propertiesProc.input)
        self.compositor.setOverlayProperties(colormap=colormap)
        for img in self.imagesDir.glob('*.png'):
            mask = gutils.cvImread_rgb(self.labelMasksDir/img.name, cv.IMREAD_UNCHANGED)
            outputFile = self.overlaysDir / img.with_suffix('.jpg').name
            self.overlayMaskOnImage(img, mask, labels, outputFile)
        self.compositor.setOverlayProperties(**oldProps)

    def overlayMaskOnImage(self, image, mask, labelMap=None, outputFile=None):
        compositor = self.compositor
        compositor.clearOverlays()
        colors = np.unique(mask.ravel())
        if labelMap is not None:
            names = labelMap[colors[colors > 0]]
        else:
            names = None
        compositor.setImage(image)
        compositor.addLabelMask(mask, names)
        if outputFile is not None:
            compositor.save(outputFile)
        return compositor

    def __reduce__(self):
        return PngExportWorkflow, (self.name, self.localFolder)
