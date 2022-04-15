from __future__ import annotations

import math
import os
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from sklearn.model_selection import train_test_split
from utilitys import fns
from utilitys.typeoverloads import FilePath

from . import constants
from .png import PngExportWorkflow
from .utils import WorkflowDir, RegisteredPath, AliasedMaskResolver, getLinkFunc

_defaultMaskColors = (None, "binary", constants.DEFAULT_RGB_CMAP)


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
        maskColors=_defaultMaskColors,
        maxNumericLabel: int = None,
        treatAsCache=False,
    ):
        if not len(labelMaskFiles):
            return
        if outputNames is None:
            outputNames = labelMaskFiles
        outputNames = np.array([os.path.basename(name) for name in outputNames])

        if treatAsCache:
            for subdir in self.labelMasksDir, self.binaryMasksDir, self.rgbMasksDir:
                self._getNewAndDeleteUnusedImages(outputNames, subdir)
        cmapDirMapping = {
            **{color: self.rgbMasksDir for color in pg.colormap.listMaps()},
            None: self.labelMasksDir,
            "binary": self.binaryMasksDir,
        }

        outputToMaskNameMap = dict(zip(outputNames, labelMaskFiles))
        # Force "None" to be the first colormap of the set to ensure that
        # file copying takes place when there is no alias mapping.
        # This is much faster in cases where the image data never needs to be read
        maskColors = [None] + [
            m for m in maskColors if m is not None and m in cmapDirMapping
        ]
        for outputName, inputMaskOrName in outputToMaskNameMap.items():
            for cmap in maskColors:
                dir_ = cmapDirMapping[cmap]
                fullOutputName = dir_.joinpath(outputName)
                # Delay reading input for as long as possible
                inputMaskOrName = self._resolveInputAndExportSingleMask(
                    inputMaskOrName, fullOutputName, cmap, resolver, maxNumericLabel
                )

    @staticmethod
    def _resolveInputAndExportSingleMask(
        inputMaskOrName, outputName, cmap, resolver, maxNumericLabel
    ):
        if os.path.exists(outputName):
            return inputMaskOrName
        if (
            (not resolver.hasClassInfo)
            and cmap is None
            and isinstance(inputMaskOrName, FilePath.__args__)
        ):
            # Special case: can copy output to input
            getLinkFunc()(inputMaskOrName, outputName)
            return inputMaskOrName
        # else reading the label image is required, try adding class info
        # from all labels so coloring is consistent
        numClasses = maxNumericLabel + 1
        if resolver.hasClassInfo:
            numClasses = resolver.numClasses
        inputMaskOrName = resolver.getMaybeResolve(inputMaskOrName)
        resolver.generateColoredMask(
            inputMaskOrName, outputName, numClasses, cmap, resolve=False
        )
        return inputMaskOrName

    @staticmethod
    def _getNewAndDeleteUnusedImages(shouldExist: t.Sequence[str], folder: Path):
        existing = {im.name for im in folder.glob("*.png")}
        shouldExist = set(shouldExist)
        for file in existing.difference(shouldExist):
            # These already exist, but shouldn't under the present mapping
            os.unlink(os.path.join(folder, file))
        return shouldExist.difference(existing)


class TrainValidateTestSplitWorkflow(WorkflowDir):
    resolver: AliasedMaskResolver
    maxNumericLabel: int

    TRAIN_NAME = "train"
    VALIDATION_NAME = "val"
    TEST_NAME = "test"

    trainDir = RegisteredPath()
    validateDir = RegisteredPath()
    testDir = RegisteredPath()

    filteredSummaryFile = RegisteredPath(".csv")
    classInfoFile = RegisteredPath(".csv")

    def runWorkflow(
        self,
        labelMap: pd.DataFrame | str | Path = None,
        balanceClasses=True,
        balanceFunc="median",
        valPct=0.15,
        testPct=0.15,
        replace=False,
        testOnUnused=True,
        maxTestSamps=None,
        maskColors=_defaultMaskColors,
    ):
        """
        From a set of label and image files, forms train, validate, and test subsets.

        :param labelMap: Dataframe with numeric_label and label columns. Matches a class label against its numeric
          mask value. One class is allowed to have multiple numeric labels. If not provided, it will default to
          ``ComponentImagesWorkflow.allLabelsFile``.
        :param balanceClasses: If *True*, ``balanceFunc`` is called on the number of items in each class to subsample
          overrepresented data.
        :param balanceFunc: See ``balanceClasses`` description. Disabled if that is *False*.
        :param valPct: Fraction (between 0 and 1) of validation data
        :param testPct: Fraction (between 0 and 1) of test/holdout data
        :param replace: If *True*, sampling of validation and test data occurs with replacement
        :param testOnUnused: If *True*, holdout data can originate from data that was discarded either through class
          exclusion or class balancing.
        :param maxTestSamps: Max number of holdout samples. If *None*, no limit is enforced.
        :param maskColors: Mask types to generate. *None* is the default label mask (strongly recommended),
          "binary" is for a black-white human-visible export, and any recognizable color ("viridis", "magma",
          etc.) will make an rgb-colored mask.
        """
        labelMap = self._resolveLabelMap(labelMap)
        exportWf = self.parent.get(PngExportWorkflow)
        if self.filteredSummaryFile.exists():
            summary = pd.read_csv(self.filteredSummaryFile, na_filter=False)
        else:
            summary = self.createGetFilteredSummaryDf(
                pd.read_csv(exportWf.summaryFile), labelMap
            )
        self.maxNumericLabel = summary["numericLabel"].max()
        self.resolver = AliasedMaskResolver(labelMap)
        if self.resolver.hasClassInfo:
            self.resolver.classInfo.to_csv(self.classInfoFile)

        datasets = []
        for dir_, typ in zip(
            [self.trainDir, self.validateDir, self.testDir],
            [self.TRAIN_NAME, self.VALIDATION_NAME, self.TEST_NAME],
        ):
            data = summary[summary["dataType"] == typ]
            datasets.append({"dir": dir_, "data": data})

        fns.mprocApply(
            self._exportDatatypePortion,
            datasets,
            extraArgs=(exportWf,),
            descr="Forming Train/Val/Test Sets",
            debug=constants.DEBUG,
        )

    def _exportDatatypePortion(self, dirAndData: dict, exportWf: PngExportWorkflow):
        PEW = PngExportWorkflow
        linkFunc = getLinkFunc()
        destDir = dirAndData["dir"]
        df = dirAndData["data"]
        maskWf = LabelMaskResolverWorkflow(destDir, createDirs=True)
        # MaskWf generates colored, normal, and binary scaled masks
        maskWf.runWorkflow(
            df["compImageFile"].apply(lambda el: exportWf.labelMasksDir / el),
            self.resolver,
            maskColors=self.input["maskColors"],
            maxNumericLabel=self.maxNumericLabel,
            treatAsCache=True,
        )

        imageDir = destDir / PEW.imagesDir
        imageDir.mkdir(exist_ok=True)
        keepImages = set(df["compImageFile"])
        existing = {im.name for im in imageDir.glob("*.png")}
        newImages = keepImages.difference(existing)
        toDelete = existing.difference(keepImages)
        for name in toDelete:
            exportWf.imagesDir.joinpath(name).unlink()
        for name in newImages:
            dst = imageDir / name
            src = exportWf.imagesDir / name
            linkFunc(src, dst)

    def createGetFilteredSummaryDf(self, summaryDf, labelInfoDf):
        filtered = self._filterByLabel(summaryDf, labelInfoDf)
        if self.input.get("balanceClasses"):
            filtered = self._balanceClasses(filtered, labelInfoDf)
        filtered = self._addTrainValTestInfo(filtered, summaryDf)

        filtered.to_csv(self.filteredSummaryFile, index=False)
        return filtered

    def _resolveLabelMap(self, labelMap: pd.DataFrame | str = None):
        if labelMap is None:
            return None
        if not isinstance(labelMap, (pd.Series, pd.DataFrame)):
            labelMap = pd.read_csv(labelMap, index_col="numeric_label")
        if "numeric_label" in labelMap.columns:
            labelMap = labelMap.set_index("numeric_label")
        return labelMap

    @staticmethod
    def _filterByLabel(summaryDf, labelInfoDf):
        if labelInfoDf is None:
            # Valid if any label is present
            membership = summaryDf["label"].notnull().to_numpy(bool)
        else:
            membership = np.isin(summaryDf["numericLabel"], labelInfoDf.index)
        # Only train on expected labels
        summaryDf = summaryDf[membership]
        return summaryDf

    def _balanceClasses(self, summaryDf, labelInfoDf=None):
        if labelInfoDf is None:
            groupCol = "label"
            # Populate 'groupCol' so outer access to this file can always use that column
            summaryDf["groupCol"] = summaryDf["label"]
        else:
            groupCol = "labelGroup"
            summaryDf = summaryDf.copy()
            summaryDf[groupCol] = labelInfoDf.loc[
                summaryDf["numericLabel"], "label"
            ].to_numpy(str)
        grouped = summaryDf.groupby(groupCol)
        sampSize = math.ceil(grouped.size().apply(self.input["balanceFunc"]))
        if self.input.get("replace"):
            sampler = lambda el: el.sample(n=sampSize, replace=True)
        else:
            sampler = lambda el: el.sample(n=min(len(el), sampSize))
        summaryDf = grouped.apply(sampler).droplevel(groupCol)
        return summaryDf

    def _addTrainValTestInfo(self, summaryDf, fullSummaryDf):
        testOnUnused = self.input.get("testOnUnused")
        unaugmented = lambda df: df[~df["imageFile"].str.contains("augmented")]
        # Just to silence intellisense; unused will be set below
        unused = None
        if testOnUnused:
            # Don't dip into training data, instead use unclassified data from the full summary
            unused = unaugmented(
                fullSummaryDf.loc[fullSummaryDf.index.difference(summaryDf.index)]
            )
            if not len(unused):
                # There is no unused data, so just use the training data as normal
                testOnUnused = False
        # Check once more for training on unused since it may have changed
        if testOnUnused:
            numSamps = min(len(unused), int(len(summaryDf) * self.input["testPct"]))
            testComps = unused.sample(n=numSamps)
            maxTest = self.input["maxTestSamps"]
            if maxTest and len(testComps) > maxTest:
                testComps = testComps.sample(n=maxTest)
            trainTemp = summaryDf.index
            summaryDf = summaryDf.append(testComps)
            testIds = testComps.index
        else:
            nonAugmentedIds = unaugmented(summaryDf).index.to_numpy()
            _, testIds = train_test_split(
                nonAugmentedIds, test_size=self.input["testPct"]
            )
            trainTemp = np.setdiff1d(summaryDf.index, testIds)
        _, valIds = train_test_split(trainTemp, test_size=self.input["valPct"])
        # By default, those which aren't test or validate will be training
        summaryDf["dataType"] = self.TRAIN_NAME
        summaryDf.loc[testIds, "dataType"] = self.TEST_NAME
        summaryDf.loc[valIds, "dataType"] = self.VALIDATION_NAME
        return summaryDf

    def createDirs(self, excludeExprs=(".",)):
        super().createDirs(excludeExprs)
        for sub in PngExportWorkflow.imagesDir, PngExportWorkflow.labelMasksDir:
            for parent in self.trainDir, self.validateDir, self.testDir:
                toCreate = parent / sub
                if any(ex in str(toCreate) for ex in excludeExprs):
                    continue
                toCreate.mkdir(exist_ok=True)
