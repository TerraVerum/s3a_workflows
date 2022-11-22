import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from qtextras import fns, widgets
from qtextras.typeoverloads import FilePath
from s3a import generalutils as gutils
from tqdm import tqdm

from . import constants
from .compimgs import ComponentImagesWorkflow
from .utils import RegisteredPath, WorkflowDirectory


class PngExportWorkflow(WorkflowDirectory):

    TRAIN_NAME = "train"
    VALIDATION_NAME = "validation"
    TEST_NAME = "test"

    ALL_DATA_TYPE_NAMES = [TRAIN_NAME, VALIDATION_NAME, TEST_NAME]

    imagesDir = RegisteredPath()
    labelMasksDir = RegisteredPath()
    overlaysDir = RegisteredPath()

    summariesDir = RegisteredPath()
    summaryFile = RegisteredPath(".csv")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compositor = widgets.MaskCompositor()
        self.compositor.propertiesProc(colormap=constants.DEFAULT_RGB_CMAP)

    def runWorkflow(
        self,
        overlayOptions=None,
    ):
        """
        Automatically generates the Neural Network data in an appropriate directory
        structure and format in the base path with the resized and padded images and
        corresponding binary Masks.

        Parameters
        ----------
        overlayOptions
            If *None*, no overlays are created. Otherwise, this is a dict of options
            for overlaying. Can include ``opacity`` between 0->1 and ``colormap``
            matching pyqtgraph or matplotlib colormap
        """
        compImgsWf = self.parent().get(ComponentImagesWorkflow)
        files = np.array(list(compImgsWf.compImgsDir.glob("*.*")))
        stems = [f.stem for f in files]
        if self.summaryFile.exists():
            summary = pd.read_csv(self.summaryFile)
            newFiles = fns.naturalSorted(
                files[
                    np.isin(
                        stems, [Path(f).stem for f in summary["imageFile"]], invert=True
                    )
                ]
            )
        else:
            newFiles = fns.naturalSorted(files)

        fns.multiprocessApply(
            self._exportSinglePcbImage,
            newFiles,
            descr="Exporting Png Files",
            showProgress=True,
            applyAsync=True,
            debug=constants.DEBUG,
        )

        self.createMergedSummaries()
        if overlayOptions is not None:
            self.createOverlays(overlayOptions=overlayOptions, showProgress=True)

    def _exportSinglePcbImage(self, compImgsFile):
        outputSummaryName = self.summariesDir / compImgsFile.with_suffix(".csv").name
        if outputSummaryName.exists():
            return

        outDf = ComponentImagesWorkflow.readDataframe(compImgsFile)
        outDf["imageFile"] = compImgsFile.with_suffix(".png").name

        exportedImgs = []
        for index, row in outDf.iterrows():
            imageName = self._exportSingleComp(index, row)
            exportedImgs.append(imageName)
        outDf["compImageFile"] = exportedImgs
        outDf: pd.DataFrame
        outDf.drop(columns=["labelMask", "image"]).to_csv(
            outputSummaryName, index=False
        )

    def _exportSingleComp(self, index, row: dict):
        exportName = f'{os.path.splitext(row["imageFile"])[0]}_id_{index}.png'

        for retKey, dir_ in zip(
            ["image", "labelMask"], [self.imagesDir, self.labelMasksDir]
        ):
            saveName = dir_ / exportName
            gutils.cvImsaveRgb(saveName, row[retKey])

        return exportName

    def createMergedSummaries(self):
        if not list(self.summariesDir.iterdir()):
            # No summaries to concatenate
            return
        concatDf = pd.concat([pd.read_csv(f) for f in self.summariesDir.glob("*.csv")])
        concatDf.to_csv(self.summaryFile, index=False)
        return concatDf

    def createOverlays(
        self, labels: pd.Series = None, overlayOptions=None, showProgress=False
    ):
        """
        Overlays masks on top of images and saves to a new directory
        :param labels: Mapping of numeric mask value to its string label
        :param overlayOptions: Properties used for mask overlays
        :param showProgress: Whether to display a progress bar during overlay creation
        """
        if labels is None:
            labels = self.summaryFile
        overlayOptions = overlayOptions or {}
        if isinstance(labels, FilePath.__args__):
            labels = pd.read_csv(
                labels, dtype=str, na_filter=False, index_col=["numericLabel"]
            )
            labels = labels["label"].drop_duplicates()
            labels[labels.str.len() == 0] = "<blank>"
        proc = self.compositor.propertiesProc
        oldProps = {**proc.extra, **proc.parameterCache}
        self.compositor.updateLabelMap(labels)
        # Account for possible "default" colormap
        if overlayOptions.get("colormap", "default").lower() == "default":
            overlayOptions["colormap"] = constants.DEFAULT_RGB_CMAP
        self.compositor.propertiesProc(**overlayOptions)
        existingIms = {im.stem for im in self.overlaysDir.glob("*.*")}
        imgList = fns.naturalSorted(
            [im for im in self.imagesDir.glob("*.png") if im.stem not in existingIms]
        )
        if showProgress:
            imgList = tqdm(imgList, "Creating Overlays")
        for img in imgList:
            mask = gutils.cvImreadRgb(
                self.labelMasksDir / img.name, cv.IMREAD_UNCHANGED
            )
            outputFile = self.overlaysDir / img.with_suffix(".jpg").name
            self.overlayMaskOnImage(img, mask, outputFile)
        proc(**oldProps)

    def overlayMaskOnImage(self, image, mask, outputFile=None):
        compositor = self.compositor
        compositor.setImage(image)
        if compositor.image.dtype == np.uint8:
            compositor.setLevels([0, 255])
        compositor.addLabelMask(mask, clearOverlays=True)
        if outputFile is not None:
            compositor.save(outputFile)
        return compositor

    def __reduce__(self):
        return PngExportWorkflow, (self.name, self.localFolder)
