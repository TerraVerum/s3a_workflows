from __future__ import annotations

import os.path
import re
import shutil
import sys
import typing as t
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg

from s3a import generalutils as gutils
from utilitys import fns, NestedProcess, ProcessIO
from utilitys.typeoverloads import FilePath
from utilitys.processing import AtomicProcess

def defaultTitle(name, trimExprs, prefix, suffix):
    name = fns.pascalCaseToTitle(name).replace(' ', '_').lower() + suffix
    for suff in trimExprs:
        name = re.sub(f'_?{suff}', '', name)
    if isinstance(prefix, RegisteredPath):
        name = os.path.join(prefix.subPath, name)
    elif prefix:
        name = prefix + name
    return name

def titleCase(*args):
    name = defaultTitle(*args)
    return fns.pascalCaseToTitle(name)

class RegisteredPath:

    def __init__(self,
                 suffix='',
                 prefix: str | RegisteredPath=None,
                 trimExprs=('file', 'path', 'dir'),
                 title: t.Callable=defaultTitle,
                 output=True
                 ):
        self.prefix = prefix
        self.suffix = suffix
        self.trimExprs = trimExprs
        self.output = output
        self.title = title

    def __set_name__(self, owner, name):
        # See utilitys.misc.DeferredActionStackMixin for description of why copy() is used
        name = self.title(name, self.trimExprs, self.prefix, self.suffix)
        self.subPath = name
        if self.output:
            paths = owner.outputPaths.copy()
            paths.add(name)
            owner.outputPaths = paths

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return obj.workflowDir / self.subPath

    def __fspath__(self):
        return self.subPath

# Simple descriptor since @property doesn't work for classes
class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class WorkflowDir(AtomicProcess):
    outputPaths: t.Set[str] = set()
    name: str = None

    def __init__(
        self,
        folder: Path | str,
        *,
        reset=False,
        createDirs=False,
        **kwargs
    ):
        self.workflowDir = Path(folder)

        if reset:
            self.resetRegisteredPaths()
        if createDirs:
            self.createDirs()

        super().__init__(self.runWorkflow, **kwargs)

    def resetRegisteredPaths(self):
        # Sort so parent paths are deleted first during rmtree
        for path in sorted(self.outputPaths):
            path = self.workflowDir / path
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def createDirs(self, excludeExprs=('.',)):
        # Sort so parent paths are created first
        self.workflowDir.mkdir(exist_ok=True)
        for path in sorted(self.outputPaths):
            if any(expr in path for expr in excludeExprs):
                continue
            self.workflowDir.joinpath(path).mkdir(exist_ok=True)

    def runWorkflow(self, *args, **kwargs):
        pass

T = t.TypeVar('T')

class NestedWorkflow(NestedProcess):
    stages: list[WorkflowDir]

    def _stageSummaryWidget(self):
        pass

    def __init__(
        self,
        folder: FilePath,
        name: str = None,
        createDirs=False,
        reset=False
    ):
        super().__init__(name)
        self.workflowDir = folder
        self.stageCounter = 1
        if reset:
            self.resetRegisteredPaths()
        if createDirs:
            self.workflowDir.mkdir(exist_ok=True)
            self.createDirs()

    def run(self, io: ProcessIO = None, disable=False, **runKwargs):
        runKwargs.update(parent=self)
        return super().run(io, disable, **runKwargs)

    def addFunction(self, workflowClass: t.Type[T], **kwargs) -> T:
        basePath = self.workflowDir
        baseName = workflowClass.name or workflowClass.__name__
        kwargs.setdefault('name', f'{self.stageCounter}. {fns.pascalCaseToTitle(baseName.replace("Workflow", ""))}')
        kwargs.setdefault('interactive', False)
        folder = basePath / kwargs['name']
        wf = workflowClass(folder, **kwargs)
        self.stages.append(wf)
        self.stageCounter += 1
        return wf

    # Alias for readability
    addWorkflow = addFunction

    def reflectDirChange(self):
        for wf in self.stages:
            wf.workflowDir = self.workflowDir / wf.workflowDir.name

    def resetRegisteredPaths(self):
        for wf in self.stages:
            wf.resetRegisteredPaths()

    def createDirs(self, excludeExprs=('.',)):
        for wf in self.stages:
            wf.createDirs(excludeExprs)

    def get(self, wfClass: t.Type[T]) -> T:
        for stage in self.stages:
            if isinstance(stage, wfClass):
                return stage
        # Requested stage type is not present
        raise KeyError(f'Workflow type "{wfClass}" is not present')


class AliasedMaskResolver:
    classInfo: pd.DataFrame

    def __init__(
        self,
        numberLabelMap=None,
        labelMasksDir=None,
    ):
        """
        :param numberLabelMap: pd.Series
            Tracks the relationship between numeric mask values and class labels. Its index is the numeric mask value
            and the value is the class label (can also be numeric)
        """
        self.masksDir = labelMasksDir or Path()

        self.hasClassInfo = numberLabelMap is not None
        if not self.hasClassInfo:
            # Accept all labels as unique
            numberLabelMap = np.arange(np.iinfo('uint16').max)
        self.setNumberLabelMap(numberLabelMap)

    def setNumberLabelMap(self, numberLabelMap):
        if not isinstance(numberLabelMap, pd.Series):
            numberLabelMap = pd.Series(numberLabelMap, numberLabelMap)
        self.classInfo = self._createOutputClassMapping(numberLabelMap)
        self.hasClassInfo = True

    def setClassInfo(self, classInfoDf: pd.DataFrame):
        if 'numeric_label' in classInfoDf:
            classInfoDf = classInfoDf.set_index('numeric_label')
        self.classInfo = classInfoDf
        self.hasClassInfo = True

    @staticmethod
    def _createOutputClassMapping(outputClasses: pd.Series):
        """
        Resolves potential aliases in the output class mapping for unambiguous mask value to class number matching
        """
        classes, numericLabels = np.unique(outputClasses.to_numpy(str), return_inverse=True)
        outputDf = pd.DataFrame()
        outputDf['label'] = outputClasses
        outputDf['numeric_class'] = numericLabels + 1
        # Add in background for easy indexing
        outputDf.index.name = 'numeric_label'
        outputDf.loc[0, ['numeric_class', 'label']] = (0, 'BGND')
        # Type not preserved when adding new row for some reason?
        outputDf['numeric_class'] = outputDf['numeric_class'].astype(int)
        return outputDf

    @property
    def numClasses(self):
        return len(self.classInfo['numeric_class'].unique()) if self.hasClassInfo else None

    def generateColoredMask(
        self,
        labelMask: np.ndarray | FilePath,
        outputFile,
        numClasses=None,
        colorMap: str=None,
        resolve=True
    ):
        """
        A helper function that generates rescaled or RGB versions of label masks
        :param labelMask: numpy image to transform (or input file containing a mask)
        :param outputFile: Location to export the transformed image
        :param colorMap: A string of the Matplotlib color map to use for the generated RGB ground truth segmentation masks.
            Acceptable color maps are restrained to the following:
             https://matplotlib.org/stable/tutorials/colors/colormaps.html. If *None*, uses the raw label mask without
             any changes. If "binary", turns any pixel > 0 to white and any pixel = 0 as black
        :param numClasses: Total number of classes for scaling rgb values. If *None*, uses the number of classes
            present in the image. Can also be set to ``self.numClasses`` for global reference.
        :param resolve: Whether the mask should be resolved (i.e. does it contain numeric aliases?) or not (i.e.
          was it already resolved from a previous call to ``self.get()``?)
        """
        labelMask = self.getMaybeResolve(labelMask, resolve=resolve)
        if colorMap is None:
            gutils.cvImsave_rgb(outputFile, labelMask)
            return
        if colorMap == 'binary':
            # No need to use colormap -- just force high values on save
            # Values > 1 should all clip to 255
            gutils.cvImsave_rgb(outputFile, (labelMask > 0).astype('uint8') * 255)
            return
        if numClasses is None:
            numClasses = np.max(labelMask)
        allColors = pg.colormap.get(colorMap).getLookupTable(nPts=numClasses)
        item = pg.ImageItem(labelMask, levels=[0, numClasses])
        item.setLookupTable(allColors, update=True)
        item.save(str(outputFile))

    def getMaybeResolve(self, mask: FilePath | np.ndarray, resolve=True):
        if isinstance(mask, FilePath.__args__):
            mask = gutils.cvImread_rgb(self.masksDir/mask, cv.IMREAD_UNCHANGED)
        # Only keep known labels
        if not resolve:
            return mask
        mask[(mask > 0) & ~np.isin(mask, self.classInfo.index)] = 0
        # Account for aliasing
        # Ravel since pandas doesn't like 2d indexing to broadcast
        mask[mask > 0] = self.classInfo.loc[mask[mask > 0], 'numeric_class'].to_numpy(int)
        return mask

# Override the Parallel class since there's no easy way to provide more informative print messages
try:
    import joblib
    class NamedParallel(joblib.Parallel):
        def __init__(self, *args, name=None, **kwargs):
            super().__init__(*args, **kwargs)
            if name is None:
                name = str(self)
            self.name = name

        def _print(self, msg, msg_args):
            if not self.verbose:
                return
            if self.verbose < 50:
                writer = sys.stderr.write
            else:
                writer = sys.stdout.write
            msg = msg % msg_args
            writer('[%s]: %s\n' % (self.name, msg))
except ImportError:
    # Joblib not available
    pass


def stringifyDict(item):
    if isinstance(item, dict):
        for kk, vv in item.items():
            item[kk] = stringifyDict(vv)
    elif isinstance(item, tuple):
        item = tuple(stringifyDict(list(item)))
    elif isinstance(item, list):
        for ii, el in enumerate(item):
            item[ii] = stringifyDict(el)
    elif not isinstance(item, (int, float, bool, str, type(None))):
        item = str(item)
    return item
