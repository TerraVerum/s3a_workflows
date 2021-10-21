from __future__ import annotations

import typing as t
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from s3a import ComponentIO, REQD_TBL_FIELDS as RTF, ComplexXYVertices, XYVertices, TableData
from s3a.generalutils import pd_iterdict
from utilitys import fns, PrjParam
from utilitys.typeoverloads import FilePath

from . import constants
from .constants import RNG
from .fmtinput import FormattedInputWorkflow
from .utils import WorkflowDir, RegisteredPath

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
      clipToBbox=True
    ):
        if bbox is None:
            bbox = np.array([[0, 0], [2000, 2000]])
        span = bbox.ptp(0)
        if sizes is None:
            sizes = np.abs(RNG.normal(0.05*span, 0.05*span, (numRows, 2)))
        if centers is None:
            centers = RNG.uniform(bbox[0], bbox[1], (numRows, 2))
        if rotations is None:
            rotations = np.zeros(numRows)
            changeIdxs = RNG.choice(numRows, int(numRows * rotationPct), replace=False)
            rotations[changeIdxs] = RNG.normal(0, 2, len(changeIdxs))
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


class ComponentImagesWorkflow(WorkflowDir):
    """
    Prescribes a workflow for generating data sufficient for SMD label-based segmentation unit.
    The final output is a pickled dataframe of component images as well as a populated directory with
    most intermediate steps, and some fitted data transformers. See `run` for more information
    """
    compImgsDir = RegisteredPath()
    allLabelsFile = RegisteredPath('.csv')
    fullLabelMasksDir = RegisteredPath()
    compImgsFile = RegisteredPath('.pkl')

    labelField: PrjParam | str = None

    def __init__(
      self,
      workflowFolder: Path | str,
      config: dict = None,
      **kwargs
    ):
        """
        Initializes the workflow with a location and set of data parameters
        """
        super().__init__(workflowFolder, config=config, **kwargs)
        self.io = ComponentIO()
        self.augmentor = ComponentGenerator()

    @classmethod
    def readDataframe(cls, file):
        """
        Since files don't have a zip extension, the default call to `read_pickle` will fail. This lets other
        classes become agnostic about adding a "compression" parameter
        """
        return pd.read_pickle(file, compression='zip')


    def createGetLabelMapping(self):
        """
        Creates a complete list of all labels from the cleaned input
        """
        if self.allLabelsFile.exists():
            return pd.read_csv(self.allLabelsFile, index_col='numeric_label', na_filter=False)
        imagesPath = Path(self.input['imagesPath'])
        labelsSer = pd.concat(
            [
                pd.read_csv(f, dtype=str, na_filter=False)[str(self.labelField)]
                for f in self.input['parent'].get(FormattedInputWorkflow).formattedFiles
                if len(list(imagesPath.glob(f.stem + '*')))
            ]
        )
        limitsCounts = labelsSer.value_counts()

        infoDf = pd.DataFrame(np.c_[limitsCounts.index, limitsCounts.values], columns=['label', 'count'])
        infoDf.index.name = 'numeric_label'
        infoDf.index += 1
        infoDf.to_csv(self.allLabelsFile)
        return infoDf['label']

    def createCompImgsDfSingle(self, file, srcDir, returnDf=False, resizeOpts=None):
        """
        Turns a csv annotation of a single image into a dataframe of cropped components from that image
        """
        srcDir = Path(srcDir)
        csvDf = self.io.importCsv(file)
        mapping = self.createGetLabelMapping()
        csvDf.columns[csvDf.columns.get_loc(self.labelField)].opts['limits'] = mapping.to_list()
        if resizeOpts is None or 'shape' not in resizeOpts:
            raise ValueError('Must pass at least a dictionary with `shape` info when creating component image exports')
        imageFiles = list(srcDir.glob(file.stem + '*'))
        if not len(imageFiles):
            warnings.warn(f'No image for {file.name}, no export will be formed', UserWarning, stacklevel=2)
            return
        imageFile = imageFiles[0]
        stats = Image.open(imageFile)
        labelMask, numericMapping = self.io.exportLblPng(
            csvDf,
            self.fullLabelMasksDir.joinpath(file.stem + '.png'),
            labelField=self.labelField,
            imageShape=(stats.height, stats.width),
            returnLabelMapping=True,
            checkVerts=False
        )

        kwargs = dict(
            srcDir=srcDir,
            labelMaskDir={imageFile.name: labelMask},
            labelMapping=numericMapping,
            resizeOpts=self.input.get('resizeOpts', {}),
            labelField=self.labelField,
            returnStats=True
        )

        originalExport = self._finalizeSingleExport(csvDf, imageFile.stem, mapping, kwargs)
        augmented = self._maybeCreateAugmentations(
            csvDf[RTF.VERTICES],
            imageFile.name,
            (stats.height, stats.width),
        )
        if augmented is not None:
            self._finalizeSingleExport(augmented, imageFile.stem + '_augmented', mapping, kwargs, prioritizeById=True)

        if returnDf:
            return originalExport

    def _maybeCreateAugmentations(self, originalVerts, imageFile, imageSize):
        augmentOpts = (self.input.get('augmentationOpts') or {}).copy()
        if not augmentOpts:
            return None
        numComps = len(originalVerts)
        numComps = int(numComps * augmentOpts.pop('fraction'))
        existingBoxes = np.array([[(s := v.stack()).min(0), s.max(0)] for v in originalVerts])
        existingSizes = existingBoxes.ptp(1)
        augSizes = RNG.normal(existingSizes.mean(0), 2 * existingSizes.std(0), (numComps, 2))
        augCenters = existingBoxes.mean(1)
        augCenters += RNG.normal(0, augCenters.std(0) / 4, augCenters.shape)
        augmentedComps = self.augmentor.makeCompDf(
            numComps,
            **augmentOpts,
            sizes=augSizes,
            centers=augCenters,
            bbox=np.array([[0, 0], imageSize[::-1]], dtype=int),
        )
        augmentedComps[RTF.IMG_FILE] = imageFile
        return augmentedComps

    def _finalizeSingleExport(self, df, name, mapping, kwargs, **extraKwargs):
        exported = self.io.exportCompImgsDf(df, **kwargs, **extraKwargs)
        idxs = [np.flatnonzero(mapping == lbl)[0] for lbl in exported['label']]
        exported['numericLabel'] = mapping.index[idxs]
        if self.input['forceVerticalOrientation']:
            exported = self.maybeReorientCompImgs(exported, df[RTF.VERTICES])
        # Unjumble row ordering
        colOrder = ['instanceId', 'label', 'numericLabel', 'offset', 'rotation', 'image', 'labelMask']
        # Ensure everything requested is present
        assert all(c in exported for c in colOrder)
        exported = exported[colOrder]
        outputName = (self.compImgsDir / name).with_suffix('.pkl')
        compressionOptions = {"method": "zip", "archive_name": outputName.name}
        exported.to_pickle(outputName, compression=compressionOptions)
        return exported

    def maybeReorientCompImgs(
      self,
      df: pd.DataFrame,
      vertices: t.Sequence[ComplexXYVertices],
    ):
        """
        Ensures the width of all SMD components is less than the height (i.e. every component will have a "vertical"
        alignment"
        """
        df['rotated'] = False

        for (idx, row), verts in zip(pd_iterdict(df, index=True), vertices):
            *rect, _orient = cv.minAreaRect(verts.stack())
            rotPoints = cv.boxPoints((*rect, _orient - row['rotation']))
            xySpan = np.ptp(rotPoints, axis=0)
            # If width has a larger span than height, rotate so all components have the same preferred component
            # aspect
            if xySpan[0] > xySpan[1]:
                for kk in 'image', 'labelMask':
                    df.at[idx, kk] = cv.rotate(row[kk], cv.ROTATE_90_CLOCKWISE)
                df.at[idx, 'rotation'] += 90
        return df

    def createMultipleCompImgs(self, fullImagesDir, inputFiles):
        # Save to intermediate directory first to avoid multiprocess comms bandwidth issues
        fns.mproc_apply(
            self.createCompImgsDfSingle,
            inputFiles,
            srcDir=fullImagesDir,
            resizeOpts=self.input['resizeOpts'],
            descr='Creating component image pickle files',
            debug=constants.DEBUG,
        )

    def mergeCompImgs(self):
        """
        After single image features are created, this merges the results into a single dataframe for easier reading
        later
        """
        allDfs = []
        for file in self.compImgsDir.glob('*.pkl'):
            subdf = self.readDataframe(file)
            allDfs.append(subdf)
        featsConcat = pd.concat(allDfs, ignore_index=True)
        compressionOptions = {"method": "zip", "archive_name": self.compImgsFile.name}
        featsConcat.to_pickle(self.compImgsFile, compression=compressionOptions)
        return featsConcat

    def runWorkflow(
      self,
      parent,
      imagesPath=None,
      s3aProj: FilePath | dict = None,
      labelField: PrjParam | str = None,
      tryMergingComps=False,
      resizeOpts: dict=None,
      augmentationOpts: dict=None,
      forceVerticalOrientation=False,
    ):
        """
        Entry point for creating image features
        :param parent: Parent NestedWorkflow
        :param imagesPath: Directory of full-sized PCB images
        :param s3aProj: S3A project file or dict for interpreting csvs
        :param labelField: Field to use as label moving forward
        :param tryMergingComps: Whether to attempt creating a merged dataframe of all subimage samples. This
          will likely fail if the shape is too large due to memory constraints
        :param resizeOpts: Passed to exporting images
        :param augmentationOpts: Parameters for producing subimage augmentations. If *None*, no augmentations
          will be produced
        :param forceVerticalOrientation: Whether all exported images should be aligned such that their longest axis is
          top-to-bottom of the image
        """
        if s3aProj is not None:
            if isinstance(s3aProj, FilePath.__args__):
                s3aProj = fns.attemptFileLoad(s3aProj)
            self.io.tableData.loadCfg(cfgDict=s3aProj)
            self.augmentor.tableData = self.io.tableData
        if labelField is not None:
            self.labelField = self.io.tableData.fieldFromName(labelField)
        if not self.labelField:
            raise ValueError('A label field must be selected before images can be exported')

        files = parent.get(FormattedInputWorkflow).formattedFiles

        generated = {f.stem for f in self.compImgsDir.glob('*.*')}
        newFiles = fns.naturalSorted(f for f in files if f.stem not in generated)
        if newFiles:
            # Force label file to be recreated
            self.allLabelsFile.unlink(missing_ok=True)
        self.createMultipleCompImgs(imagesPath, newFiles)

        if tryMergingComps:
            merged = self.mergeCompImgs()
            return merged
