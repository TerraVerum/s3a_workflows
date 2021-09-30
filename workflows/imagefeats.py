from __future__ import annotations

import functools
import math
import os
import pickle
import shutil
import tempfile
import typing as t
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from s3a import TableData, ComponentIO, generalutils as gutils, REQD_TBL_FIELDS as RTF, ComplexXYVertices
from s3a.compio.exporters import SerialExporter
from s3a.compio.importers import SerialImporter
from s3a.generalutils import pd_iterdict, resize_pad
from utilitys import fns, PrjParam
from utilitys.typeoverloads import FilePath
from workflows.constants import FPIC_SMDS, FPIC_IMAGES
from workflows.utils import WorkflowDir, RegisteredPath, AliasedMaskResolver, NestedWorkflow


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
            # debug=True,
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
        :param shape: Output component shape (width, height)
        :param keepAspectRatio: Whether the aspect ratio of component images should be maintained during resizing
        :param s3aProj: S3A project file or dict for interpreting csvs
        :param labelField: Field to use as label moving forward
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
                df_.at[idx, 'image'] = resize_pad(img, (50,50))
            return df_
        df = fns.readDataFrameFiles(compImgsWf.compImgsDir, readFunc)
        # Ensure classes aren't overrepresented and blanks aren't an actual class
        df: pd.DataFrame = df[df['label'].str.len() > 0]
        df = df.groupby('label').apply(lambda el: el.sample(n=min(len(el), 150), random_state=10)).reset_index(drop=True)
        featsLabels = self.getFeatsLabels(df, compImgsWf.createGetLabelMapping())
        for xformer in tqdm([PCA(), LDA()], desc='Fitting transformers'):
            self.fitSaveTransformer(xformer, featsLabels, compImgsWf.createGetLabelMapping())


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
            # debug=True,
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


class LabelMaskResolverWorkflow(WorkflowDir):
    """
    Turns masks with potentially many-to-one label mappings to sequentially numbered output values
    """
    rgbMasksDir = RegisteredPath()
    binaryMasksDir = RegisteredPath()
    labelMasksDir = RegisteredPath()

    def runWorkflow(
        self,
        labelMaskFiles: t.List[Path | np.ndarray],
        resolver: AliasedMaskResolver,
        outputNames: t.Sequence[str] = None,
    ):
        if outputNames is None:
            outputNames = labelMaskFiles
        for mask, filename in zip(labelMaskFiles, outputNames):
            mask = resolver.getMaybeResolve(mask)
            # Fetch out here to avoid fetching inside loop
            for cmap, dir_ in zip(
                [None, 'binary', 'viridis'],
                [self.labelMasksDir, self.binaryMasksDir, self.rgbMasksDir]
            ):
                filename = os.path.basename(filename)
                resolver.generateColoredMask(mask, dir_/filename, resolver.numClasses, cmap, resolve=False)


class TrainValTestWorkflow(WorkflowDir):
    name = 'Train-Val-Test Split'

    resolver = AliasedMaskResolver()

    TRAIN_NAME = 'train'
    VALIDATION_NAME = 'val'
    TEST_NAME = 'test'


    trainDir = RegisteredPath()
    validateDir = RegisteredPath()
    testDir = RegisteredPath()

    filteredSummaryFile = RegisteredPath('.csv')
    classInfoFile = RegisteredPath('.csv')

    def runWorkflow(
        self,
        parent: NestedWorkflow,
        labelMap: pd.DataFrame=None,
        balanceClasses=True,
        balanceFunc='median',
        replaceSamps=False,
        valPct=0.15,
        testPct=0.15,
        replace=False,
        testOnUnused=False,
        maxTestSamps=None
    ):
        exportWf = parent.get(PngExportWorkflow)
        fullSummary = pd.read_csv(exportWf.summaryFile)
        summary = self.createGetFilteredSummaryDf(
            fullSummary,
            labelMap
        )
        if 'numeric_label' in labelMap.columns:
            labelMap = labelMap.set_index('numeric_label')

        self.resolver = AliasedMaskResolver(labelMap['label'])
        self.resolver.classInfo.to_csv(self.classInfoFile)

        trainSet = {self.trainDir: summary[summary['dataType'] == 'train']}
        otherSet = {}
        for dir_, typ in zip([self.validateDir, self.testDir], [self.VALIDATION_NAME, self.TEST_NAME]):
            otherSet[dir_] = summary[summary['dataType'] == typ]

        fns.mproc_apply(
            self._exportDatatypePortion,
            (trainSet, otherSet),
            extraArgs=(exportWf,),
            descr='Forming Train/Val/Test Sets',
            # debug=True
        )

    def _exportDatatypePortion(self, dirSummaryMap: dict, exportWf):
        PEW = PngExportWorkflow
        linkFunc = self._getLinkFunc()
        for destDir, df in dirSummaryMap.items():
            maskWf = LabelMaskResolverWorkflow(destDir, createDirs=True)
            # MaskWf generates colored, normal, and binary scaled masks
            maskWf.runWorkflow(
                df['compImageFile'].apply(
                    lambda el: exportWf.labelMasksDir / el),
                self.resolver
            )
            imageDir = destDir / PEW.imagesDir
            imageDir.mkdir(exist_ok=True)
            for row in pd_iterdict(df):
                compFile = row['compImageFile']
                linkFunc(exportWf.imagesDir / compFile, imageDir / compFile)

    def createGetFilteredSummaryDf(self, summaryDf, labelInfoDf):
        filtered = self._filterByLabel(summaryDf, labelInfoDf)
        if self.input.get('balanceClasses'):
            filtered = self._balanceClasses(filtered, labelInfoDf)
        filtered = self._addTrainValTestInfo(filtered, summaryDf)

        filtered.to_csv(self.filteredSummaryFile, index=False)
        return filtered

    @staticmethod
    def _filterByLabel(summaryDf, labelInfoDf):
        if labelInfoDf is None:
            # Valid if any label is present
            membership = summaryDf['label'].notnull().to_numpy(bool)
        else:
            membership = np.isin(summaryDf['numericLabel'], labelInfoDf.index)
        # Only train on expected labels
        summaryDf = summaryDf[membership]
        return summaryDf

    def _balanceClasses(self, summaryDf, labelInfoDf=None):
        if labelInfoDf is None:
            groupCol = 'label'
        else:
            groupCol = 'labelGroup'
            summaryDf = summaryDf.copy()
            summaryDf[groupCol] = labelInfoDf.loc[summaryDf['numericLabel'], 'label'].to_numpy(str)
        grouped = summaryDf.groupby(groupCol)
        sampSize = math.ceil(grouped.size().apply(self.input['balanceFunc']))
        if self.input.get('replace'):
            sampler = lambda el: el.sample(n=sampSize, replace=True)
        else:
            sampler = lambda el: el.sample(n=min(len(el), sampSize))
        summaryDf = grouped.apply(sampler).droplevel(groupCol)
        return summaryDf

    def _addTrainValTestInfo(self, summaryDf, fullSummaryDf):
        testOnUnused = self.input.get('testOnUnused')
        # Just to silence intellisense; unused will be set below
        unused = None
        if testOnUnused:
            # Don't dip into training data, instead use unclassified data from the full summary
            unused = fullSummaryDf.loc[fullSummaryDf.index.difference(summaryDf.index)]
            if not len(unused):
                # There is no unused data, so just use the training data as normal
                testOnUnused = False
        # Check once more for training on unused since it may have changed
        if testOnUnused:
            testComps = unused.sample(frac=self.input['testPct'])
            maxTest = self.input['maxTestSamps']
            if maxTest and len(testComps) > maxTest:
                testComps = testComps.sample(n=maxTest)
            trainTemp = summaryDf.index
            summaryDf = summaryDf.append(testComps)
            testIds = testComps.index
        else:
            trainTemp, testIds = train_test_split(summaryDf.index.to_numpy(), test_size=self.input['testPct'])
        _, valIds = train_test_split(trainTemp, test_size=self.input['valPct'])
        # By default, those which aren't test or validate will be training
        summaryDf['dataType'] = self.TRAIN_NAME
        summaryDf.loc[testIds, 'dataType'] = self.TEST_NAME
        summaryDf.loc[valIds, 'dataType'] = self.VALIDATION_NAME
        return summaryDf

    @staticmethod
    def _getLinkFunc():
        """
        Symlinks rarely have permission by default on windows so be able to copy if needed
        """
        # Use symlinks to avoid lots of file duplication
        try:
            linkFunc = os.symlink
            with tempfile.TemporaryDirectory() as td:
                src: Path = Path(td) / 'test'
                src.touch()
                linkFunc(src, src.with_name('testlink'))
        except (PermissionError, OSError):
            linkFunc = shutil.copy
        return linkFunc


    def createDirs(self, excludeExprs=('.',)):
        super().createDirs(excludeExprs)
        for sub in PngExportWorkflow.imagesDir, PngExportWorkflow.labelMasksDir:
            for parent in self.trainDir, self.validateDir, self.testDir:
                toCreate = parent/sub
                if any(ex in str(toCreate) for ex in excludeExprs):
                    continue
                toCreate.mkdir(exist_ok=True)
