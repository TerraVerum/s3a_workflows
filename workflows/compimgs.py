from __future__ import annotations
import functools
import typing as t
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from s3a import ComponentIO, REQD_TBL_FIELDS as RTF, ComplexXYVertices
from s3a.compio.exporters import SerialExporter
from s3a.compio.importers import SerialImporter
from s3a.generalutils import pd_iterdict
from utilitys import fns, PrjParam
from utilitys.typeoverloads import FilePath

from .constants import DEBUG, FPIC_SMDS, FPIC_IMAGES
from .utils import WorkflowDir, RegisteredPath

class ComponentImagesWorkflow(WorkflowDir):
    """
    Prescribes a workflow for generating data sufficient for SMD label-based segmentation unit.
    The final output is a pickled dataframe of component images as well as a populated directory with
    most intermediate steps, and some fitted data transformers. See `run` for more information
    """
    formattedInputPath = RegisteredPath()

    compImgsDir = RegisteredPath()
    allLabelsFile = RegisteredPath('.csv')
    compImgsFile = RegisteredPath('.pkl')

    @property
    def newInputFiles(self):
        """
        Helper property to act like a list of files from the cleansed inputs
        """
        files = self.formattedInputPath.glob('*.csv')

        generated = {f.stem for f in self.compImgsDir.glob('*.*')}
        return fns.naturalSorted(f for f in files if f.stem not in generated)

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
        self.labelField = None

    def createFormattedInputs(self, annotationPath: FilePath=None):
        """
        Generates cleansed csv files from the raw input dataframe. Afterwards, saves annotations in files separated
        by image to allow multiprocessing on subsections of components
        """
        if annotationPath is None:
            return pd.DataFrame()
        if annotationPath.is_dir():
            df = fns.readDataFrameFiles(annotationPath, SerialImporter.readFile)
        else:
            df = SerialImporter.readFile(annotationPath)
        for image, subdf in df.groupby(RTF.IMG_FILE.name):
            newName = Path(image).with_suffix('.csv').name
            dest = self.formattedInputPath/newName
            if not dest.exists():
                SerialExporter.writeFile(dest, subdf, readonly=False)
        return df


    @functools.lru_cache()
    def createGetLabelMapping(self):
        """
        Creates a complete list of all labels from the cleaned input
        """
        # Use an lru cache instead of checking for file

        # Passing dataframes over multiprocessing is slower than re-reading a new file's dataframe each iteration.
        # So, just don't use this dataframe for anything other than getting unique labels
        labelsSer = pd.concat([pd.read_csv(f, dtype=str, na_filter=False)[str(self.labelField)]
                                 for f in self.formattedInputPath.glob('*.csv')]
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
        name = file.name
        df = self.io.importCsv(file)
        mapping = self.createGetLabelMapping()
        df.columns[df.columns.get_loc(self.labelField)].opts['limits'] = mapping.to_list()
        if resizeOpts is None or 'shape' not in resizeOpts:
            raise ValueError('Must pass at least a dictionary with `shape` info when creating component image exports')
        exported = self.io.exportCompImgsDf(
            df,
            srcDir=srcDir,
            resizeOpts=self.input.get('resizeOpts', {}),
            labelField=self.labelField,
        )
        idxs = [np.flatnonzero(mapping == lbl)[0] for lbl in exported['label']]
        exported['numericLabel'] = mapping.index[idxs]
        if self.input['forceVerticalOrientation']:
            self.maybeReorientCompImgs(exported, df[RTF.VERTICES])
        else:
            exported['rotated'] = False
        # Unjumble row ordering
        colOrder = ['instanceId', 'label', 'numericLabel', 'offset', 'rotated', 'image', 'labelMask']
        # Ensure nothing was lost in the reordering
        assert set(exported) == set(colOrder)
        exported = exported[colOrder]
        exported.to_pickle((self.compImgsDir / name).with_suffix('.pkl'))
        if returnDf:
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
            xySpan = np.ptp(verts.stack(), axis=0)
            # If width has a larger span than height, rotate so all components have the same preferred component
            # aspect
            if xySpan[0] > xySpan[1]:
                for kk in 'image', 'labelMask':
                    df.at[idx, kk] = cv.rotate(row[kk], cv.ROTATE_90_CLOCKWISE)
                df.at[idx, 'rotated'] = True
        return df

    def createAllCompImgs(self, fullImagesDir):
        # Save to intermediate directory first to avoid multiprocess comms bandwidth issues
        fns.mproc_apply(
            self.createCompImgsDfSingle,
            self.newInputFiles,
            srcDir=fullImagesDir,
            resizeOpts=self.input['resizeOpts'],
            descr='Creating component image pickle files',
            debug=DEBUG,
        )

    def mergeCompImgs(self):
        """
        After single image features are created, this merges the results into a single dataframe for easier reading
        later
        """
        allDfs = []
        for file in self.compImgsDir.glob('*.pkl'):
            subdf = pd.read_pickle(file)
            allDfs.append(subdf)
        featsConcat = pd.concat(allDfs, ignore_index=True)
        featsConcat.to_pickle(self.compImgsFile)
        return featsConcat

    def runWorkflow(
        self,
        annotationPath=FPIC_SMDS,
        fullImagesDir=FPIC_IMAGES,
        s3aProj: FilePath | dict = None,
        labelField: PrjParam | str = None,
        tryMergingComps=False,
        resizeOpts: dict=None,
        forceVerticalOrientation=True
    ):
        """
        Entry point for creating image features
        :param annotationPath: CSV image annotations to convert to comp imgs
        :param fullImagesDir: Directory of full-sized PCB images. If *None*, defaults to FPIC pcb_image if FPIC
          is set on the system
        :param s3aProj: S3A project file or dict for interpreting csvs
        :param labelField: Field to use as label moving forward
        :param forceVerticalOrientation: Whether all exported images should be aligned such that their longest axis is
          top-to-bottom of the image
        :param tryMergingComps: Whether to attempt creating a merged dataframe of all subimage samples. This
          will likely fail if the shape is too large due to memory constraints
        :param resizeOpts: Passed to exporting images
        """
        if s3aProj is not None:
            if isinstance(s3aProj, FilePath.__args__):
                s3aProj = fns.attemptFileLoad(s3aProj)
                self.io.tableData.loadCfg(cfgDict=s3aProj)
        if labelField is not None:
            self.labelField = self.io.tableData.fieldFromName(labelField)
        if not self.labelField:
            raise ValueError('A label field must be selected before images can be exported')
        self.createFormattedInputs(annotationPath)

        self.createAllCompImgs(fullImagesDir)

        if tryMergingComps:
            merged = self.mergeCompImgs()
            return merged
