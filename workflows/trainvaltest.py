from __future__ import annotations

import math
import os
import shutil
import tempfile
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from s3a.generalutils import pd_iterdict
from sklearn.model_selection import train_test_split
from utilitys import fns

from .constants import DEBUG
from .png import PngExportWorkflow
from .utils import WorkflowDir, RegisteredPath, AliasedMaskResolver, NestedWorkflow

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

class TrainValidateTestSplitWorkflow(WorkflowDir):
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
        testOnUnused=True,
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
            debug=DEBUG
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
