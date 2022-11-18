from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

import pandas as pd
from s3a import (
    REQD_TBL_FIELDS as RTF,
    TableData,
    ComplexXYVertices,
    XYVertices,
    compio,
    defaultIo,
)
from s3a.compio import SerialExporter, SerialImporter
from tqdm import tqdm
from qtextras import fns
from qtextras.typeoverloads import FilePath

from .constants import RNG
from .utils import WorkflowDirectory, RegisteredPath


class FormattedInputWorkflow(WorkflowDirectory):
    formattedInputsDir = RegisteredPath()
    augmentedInputsDir = RegisteredPath()

    def __init__(self, *args, **kwargs):
        """
        Initializes the workflow with a location and set of data parameters
        """
        super().__init__(*args, **kwargs)
        self.augmentor = ComponentGenerator()

    def runWorkflow(
        self,
        annotationsPath: FilePath = None,
        augmentationOptions: dict = None,
        filterExpr: str = None,
    ):
        """
        Generates cleansed csv files from the raw input dataframe. Afterwards,
        saves annotations in files separated by image to allow multiprocessing on
        subsections of components

        Parameters
        ----------
        annotationsPath
            Can either be a file or folder path. These are the annotations that will be
            processed during the workflow.
        augmentationOptions
            Parameters for producing subimage augmentations. If *None*,
            no augmentations will be produced
        filterExpr
            If speficied, this is passed to ``annotation dataframe.query`` to filter
            out unwanted samples
        """
        if annotationsPath is None:
            return pd.DataFrame()
        annotationsPath = Path(annotationsPath)
        if annotationsPath.is_dir():
            df = pd.concat(map(SerialImporter.readFile, annotationsPath.glob("*.csv")))
        else:
            df = SerialImporter.readFile(annotationsPath)
        if filterExpr is not None:
            df = df.query(filterExpr, engine="python")
        # Ensure old naming scheme is valid
        df = df.rename(columns={"Source Image Filename": RTF.IMAGE_FILE.name})
        for imageName, subdf in tqdm(
            df.groupby(RTF.IMAGE_FILE.name), "Formatting inputs"
        ):
            newName = Path(imageName).with_suffix(".csv").name
            dest = self.formattedInputsDir / newName
            if not dest.exists():
                SerialExporter.writeFile(dest, subdf, readonly=False)
            if not self.augmentedInputsDir.joinpath(newName).exists():
                self._maybeCreateAugmentations(
                    subdf[RTF.VERTICES],
                    imageName,
                )
        return df

    def _maybeCreateAugmentations(self, originalVerts, imageFile):
        augmentOpts = (self.input.get("augmentationOptions") or {}).copy()
        if not augmentOpts or np.isclose(augmentOpts.get("fraction", 0.0), 0):
            return None
        originalVerts = compio.helpers.deserialize(
            RTF.VERTICES, originalVerts, returnErrs=False
        )
        numComps = len(originalVerts)
        numComps = int(numComps * augmentOpts.pop("fraction"))
        existingBoxes = np.array(
            [[(s := v.stack()).min(0), s.max(0)] for v in originalVerts]
        )
        existingSizes = existingBoxes.ptp(1)
        augSizes = RNG.normal(
            existingSizes.mean(0), 2 * existingSizes.std(0), (numComps, 2)
        )
        augCenters = existingBoxes.mean(1)
        augCenters += RNG.normal(0, augCenters.std(0) / 4, augCenters.shape)
        # Accomodate case where more rows are requested than exist in the dataframe
        augCenters = RNG.choice(augCenters, numComps)
        augmentedComps = self.augmentor.makeCompDf(
            numComps,
            **augmentOpts,
            sizes=augSizes,
            centers=augCenters,
            clipToBbox=False,
        )
        augmentedComps[RTF.IMAGE_FILE] = imageFile
        defaultIo.exportCsv(
            augmentedComps,
            self.augmentedInputsDir / Path(imageFile).with_suffix(".csv"),
            readonly=False,
        )
        return augmentedComps

    @property
    def formattedFiles(self):
        return fns.naturalSorted(self.formattedInputsDir.glob("*.csv"))


class ComponentGenerator:
    def __init__(
        self,
        tableData=None,
    ):
        self.tableData = tableData or TableData()

    def makeCompDf(
        self,
        numRows: int,
        bbox=None,
        sizes=None,
        centers=None,
        rotations=None,
        rotationPct=0.4,
        clipToBbox=True,
    ):
        if bbox is None:
            bbox = np.array([[0, 0], [2000, 2000]])
        span = bbox.ptp(0)
        if sizes is None:
            sizes = np.abs(RNG.normal(0.05 * span, 0.05 * span, (numRows, 2)))
        if centers is None:
            centers = RNG.uniform(bbox[0], bbox[1], (numRows, 2))
        if rotations is None:
            rotations = np.zeros(numRows)
            changeIdxs = RNG.choice(numRows, int(numRows * rotationPct), replace=False)
            rotations[changeIdxs] = RNG.normal(0, 15, len(changeIdxs))
        outBoxes = []
        for size, center, rot in zip(sizes, centers, rotations):
            points = cv.boxPoints((center, size, rot))
            if clipToBbox:
                points = np.clip(points, bbox[0], bbox[1])
            verts = ComplexXYVertices([points.astype(int).view(XYVertices)])
            outBoxes.append(verts)
        out = self.tableData.makeCompDf(numRows, sequentialIds=True)
        out[RTF.VERTICES] = outBoxes
        return out
