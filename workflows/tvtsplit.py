from __future__ import annotations

import math
import os
import shutil
import tempfile
import typing as t
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utilitys import fns

from . import constants, ComponentImagesWorkflow
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
      treatAsCache=False
    ):
        if outputNames is None:
            outputNames = labelMaskFiles
        outputNames = np.array([os.path.basename(name) for name in outputNames])

        if treatAsCache:
            newFiles = set()
            for subdir in self.labelMasksDir, self.binaryMasksDir, self.rgbMasksDir:
                newFiles.update(self._getNewAndDeleteUnusedImages(outputNames, subdir))
            membership = np.isin(outputNames, np.array(list(newFiles)))
            labelMaskFiles = np.array(labelMaskFiles, dtype=object)[membership]
            outputNames = outputNames[membership]

        for mask, filename in zip(labelMaskFiles, outputNames):
            mask = resolver.getMaybeResolve(mask)
            # Fetch out here to avoid fetching inside loop
            for cmap, dir_ in zip(
              [None, 'binary', 'viridis'],
              [self.labelMasksDir, self.binaryMasksDir, self.rgbMasksDir]
            ):
                resolver.generateColoredMask(mask, dir_/filename, resolver.numClasses, cmap, resolve=False)

    @staticmethod
    def _getNewAndDeleteUnusedImages(shouldExist: t.Sequence[str], folder: Path):
        existing = np.array([im.name for im in folder.glob('*.png')])
        for file in np.setdiff1d(existing, shouldExist):
            # These already exist, but shouldn't under the present mapping
            os.unlink(os.path.join(folder, file))
        return np.setdiff1d(shouldExist, existing)

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
      valPct=0.15,
      testPct=0.15,
      replace=False,
      testOnUnused=True,
      maxTestSamps=None
    ):
        labelMap = self._resolveLabelMap(labelMap)
        exportWf = parent.get(PngExportWorkflow)
        fullSummary = pd.read_csv(exportWf.summaryFile)
        if self.filteredSummaryFile.exists():
            summary = pd.read_csv(self.filteredSummaryFile, na_filter=False)
        else:
            summary = self.createGetFilteredSummaryDf(
                fullSummary,
                labelMap
            )
        self.resolver = AliasedMaskResolver(labelMap['label'])
        self.resolver.classInfo.to_csv(self.classInfoFile)

        datasets = []
        for dir_, typ in zip([self.trainDir, self.validateDir, self.testDir], [self.TRAIN_NAME, self.VALIDATION_NAME, self.TEST_NAME]):
            data = summary[summary['dataType'] == typ]
            datasets.append(
                {'dir': dir_, 'data': data}
            )

        fns.mproc_apply(
            self._exportDatatypePortion,
            datasets,
            extraArgs=(exportWf,),
            descr='Forming Train/Val/Test Sets',
            debug=constants.DEBUG
        )

    def _exportDatatypePortion(self, dirAndData: dict, exportWf: PngExportWorkflow):
        PEW = PngExportWorkflow
        linkFunc = self._getLinkFunc()
        destDir = dirAndData['dir']
        df = dirAndData['data']
        maskWf = LabelMaskResolverWorkflow(destDir, createDirs=True)
        # MaskWf generates colored, normal, and binary scaled masks
        maskWf.runWorkflow(
            df['compImageFile'].apply(
                lambda el: exportWf.labelMasksDir / el
            ),
            self.resolver,
            treatAsCache=True
        )

        imageDir = destDir / PEW.imagesDir
        imageDir.mkdir(exist_ok=True)
        keepImages = df['compImageFile'].to_numpy()
        existing = np.array([im.name for im in imageDir.glob('*.png')], dtype=object)
        newImages = np.setdiff1d(keepImages, existing)
        toDelete = np.setdiff1d(existing, keepImages)
        for name in toDelete:
            exportWf.imagesDir.joinpath(name).unlink()
        for name in newImages:
            dst = imageDir / name
            src = exportWf.imagesDir / name
            linkFunc(src, dst)

    def createGetFilteredSummaryDf(self, summaryDf, labelInfoDf):
        filtered = self._filterByLabel(summaryDf, labelInfoDf)
        if self.input.get('balanceClasses'):
            filtered = self._balanceClasses(filtered, labelInfoDf)
        filtered = self._addTrainValTestInfo(filtered, summaryDf)

        filtered.to_csv(self.filteredSummaryFile, index=False)
        return filtered

    def _resolveLabelMap(self, labelMap: pd.DataFrame | str=None):
        if labelMap is None:
            # warnings.warn(f'Since labelMap is *None*, "{self.name}" will default to using all raw labels.', UserWarning)
            labelMap = self.input['parent'].get(ComponentImagesWorkflow).allLabelsFile
        if not isinstance(labelMap, (pd.Series, pd.DataFrame)):
            labelMap = pd.read_csv(labelMap, index_col='numeric_label')
        if 'numeric_label' in labelMap.columns:
            labelMap = labelMap.set_index('numeric_label')
        return labelMap

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
        unaugmented = lambda df: df[~df['imageFile'].str.contains('augmented')]
        # Just to silence intellisense; unused will be set below
        unused = None
        if testOnUnused:
            # Don't dip into training data, instead use unclassified data from the full summary
            unused = unaugmented(fullSummaryDf.loc[fullSummaryDf.index.difference(summaryDf.index)])
            if not len(unused):
                # There is no unused data, so just use the training data as normal
                testOnUnused = False
        # Check once more for training on unused since it may have changed
        if testOnUnused:
            numSamps = min(len(unused), int(len(summaryDf)*self.input['testPct']))
            testComps = unused.sample(n=numSamps)
            maxTest = self.input['maxTestSamps']
            if maxTest and len(testComps) > maxTest:
                testComps = testComps.sample(n=maxTest)
            trainTemp = summaryDf.index
            summaryDf = summaryDf.append(testComps)
            testIds = testComps.index
        else:
            nonAugmentedIds = unaugmented(summaryDf).index.to_numpy()
            _, testIds = train_test_split(nonAugmentedIds, test_size=self.input['testPct'])
            trainTemp = np.setdiff1d(summaryDf.index, testIds)
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
