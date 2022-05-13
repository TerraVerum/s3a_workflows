from __future__ import annotations

import argparse
import functools
import inspect
import os.path
import re
import shutil
import tempfile
import typing as t
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from s3a import generalutils as gutils
from s3a.parameditors.algcollection import AlgParamEditor
from utilitys import NestedProcess, ProcessStage, PrjParam
from utilitys import fns, AtomicProcess
from utilitys.typeoverloads import FilePath

T = t.TypeVar("T")


def defaultTitle(name, trimExprs, prefix, suffix):
    name = fns.pascalCaseToTitle(name).replace(" ", "_").lower() + suffix
    for suff in trimExprs:
        name = re.sub(f"_?{suff}", "", name)
    if isinstance(prefix, RegisteredPath):
        name = os.path.join(prefix.subPath, name)
    elif prefix:
        name = prefix + name
    return name


def titleCase(*args):
    name = defaultTitle(*args)
    return fns.pascalCaseToTitle(name)


# Credit: https://stackoverflow.com/a/37095733/9463643
def path_is_parent(parent_path, child_path):
    # Smooth out relative path names, note: if you are concerned about symbolic links, you should use os.path.realpath too
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    # Compare the common path of the parent and child path with the common path of just the parent path. Using the commonpath method on just the parent path will regularise the path name in the same way as the comparison that deals with both paths, removing any trailing path separator
    return parent_path == os.path.commonpath([parent_path, child_path])


class RegisteredPath:
    def __init__(
        self,
        suffix="",
        prefix: str | RegisteredPath = None,
        trimExprs=("file", "path", "dir"),
        title: t.Callable = defaultTitle,
        output=True,
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


class WorkflowMixin:
    name: str = None

    def __init__(self, folder: Path | str = None, parent: NestedWorkflow = None):
        self.localFolder = Path(folder or "")
        self.parent = parent

    @property
    def workflowDir(self):
        if self.parent:
            return self.parent.workflowDir / self.localFolder
        return self.localFolder

    def resetRegisteredPaths(self):
        ...

    def createDirs(self):
        ...


class WorkflowDir(AtomicProcess, WorkflowMixin):
    outputPaths: t.Set[str] = set()

    def __init__(
        self,
        name: str = None,
        folder: Path | str = None,
        *,
        reset=False,
        createDirs=False,
        **kwargs,
    ):
        baseName = self.name or type(self).__name__.replace("Workflow", "")
        defaultName = fns.pascalCaseToTitle(baseName)
        name = name or defaultName

        if folder is None:
            folder = name

        kwargs.setdefault("interactive", False)
        AtomicProcess.__init__(self, self.runWorkflow, name=name, **kwargs)
        WorkflowMixin.__init__(self, folder)

        if reset:
            self.resetRegisteredPaths()
        if createDirs:
            self.createDirs()

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

    def createDirs(self, excludeExprs=(".",)):
        # Sort so parent paths are created first
        self.workflowDir.mkdir(exist_ok=True)
        for path in sorted(self.outputPaths):
            if any(expr in path for expr in excludeExprs):
                continue
            self.workflowDir.joinpath(path).mkdir(exist_ok=True)

    def runWorkflow(self, *args, **kwargs):
        pass


class NestedWorkflow(NestedProcess, WorkflowMixin):
    stages: list[WorkflowDir]

    def _stageSummaryWidget(self):
        pass

    def __init__(
        self,
        name: str = None,
        folder: FilePath = None,
        createDirs=False,
        reset=False,
    ):
        NestedProcess.__init__(self, name or "<Unnamed>")
        WorkflowMixin.__init__(self, folder or name)

        if reset:
            self.resetRegisteredPaths()
        if createDirs:
            self.createDirs()

    def disableStages(self, *stageClasses: t.Type[WorkflowDir]):
        for cls in stageClasses:
            self.get(cls).disabled = True

    def saveStringifiedConfig(self, folder: str | Path = None, **initKwargs):
        state = self.saveState(includeDefaults=True)
        # Make a dummy process for input parameters just to easily save its state
        initState = AtomicProcess(
            self.__init__, name="Initialization", interactive=False, **initKwargs
        ).saveState(includeDefaults=True)
        state[self.name].insert(0, initState)

        # Some values are unrepresentable in their natural form (e.g. Paths)
        state = stringifyDict(state)
        if folder is None:
            folder = self.workflowDir
        fns.saveToFile(state, Path(folder).joinpath("config.yml"))
        return state

    @classmethod
    def splitInitAndRunKwargs(cls, kwargs):
        """
        Converts a dict of potentially both __init__ keywords and run() keywords into two separate dicts
        """
        initSpec = set(inspect.signature(cls.__init__).parameters)
        initKwargs = {}
        for kw in initSpec:
            if kw in kwargs:
                initKwargs[kw] = kwargs.pop(kw)
        return initKwargs, kwargs

    def addWorkflow(self, workflowClass: t.Type[T], **kwargs) -> T:
        wf = workflowClass(**kwargs)
        return self.addProcess(wf)

    def addProcess(self, process: ProcessStage):
        if isinstance(process, Workflow_T):
            process.parent = self
            # Ensure workflow path is relative to self
        return super().addProcess(process)

    def resetRegisteredPaths(self):
        for wf in self.stages:
            if not wf.disabled:
                wf.resetRegisteredPaths()

    def createDirs(self, excludeExprs=(".",)):
        self.workflowDir.mkdir(exist_ok=True)
        for wf in self.stages:
            if not wf.disabled and isinstance(wf, Workflow_T):
                wf.createDirs(excludeExprs)

    def _getFromRoot(self, root: Workflow_T, wfClassorName):
        # TODO: Prefer stages closer to requesting process
        if isinstance(root, wfClassorName) or root.name == wfClassorName:
            return root
        for stage in root:
            if ret := self._getFromRoot(stage, wfClassorName):
                return ret
        # No matches anywhere in the tree
        return None

    def get(self, wfClass: t.Type[T] | str, missingOk=True) -> T:
        root = self
        while root.parent is not None:
            root = root.parent
        if match := self._getFromRoot(root, wfClass):
            return match
        # Stage not present already; add at self's level
        if missingOk and not isinstance(wfClass, str):
            stage = self.addWorkflow(wfClass)
            stage.disabled = True
            return stage
        # Requested stage type is not present
        raise KeyError(f'Workflow type "{wfClass}" is not present')


Workflow_T = (WorkflowDir, NestedWorkflow)


def argparseHelpAction(nested: NestedWorkflow):
    class NestedWorkflowHelp(argparse.Action):
        def __init__(self, **kwargs):
            kwargs.update(nargs=0)
            super().__init__(**kwargs)

        def __call__(self, parser, *args, **kwargs) -> None:
            state = {}
            for stage in nested.stagesFlattened:
                pgDict = fns.funcToParamDict(stage.func)
                for child in pgDict["children"]:
                    state[child["name"]] = child
            newCli = fns.makeCli(
                nested.__init__,
                convertArgs=False,
                parserKwargs=dict(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter
                ),
                **state,
            )
            newCli.print_help()
            parser.exit()

    return NestedWorkflowHelp


class AliasedMaskResolver:
    classInfo: t.Optional[pd.DataFrame]

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
        self.classInfo = None
        if self.hasClassInfo:
            # Also sets classInfo
            self.setNumberLabelMap(numberLabelMap)

    def setNumberLabelMap(self, numberLabelMap):
        if isinstance(numberLabelMap, pd.DataFrame):
            if "label" in numberLabelMap:
                numberLabelMap = numberLabelMap["label"]
            else:
                raise ValueError(
                    "Setting label map with dataframe must have a 'label' column"
                )
        if not isinstance(numberLabelMap, pd.Series):
            numberLabelMap = pd.Series(numberLabelMap, numberLabelMap)
        self.classInfo = self._createOutputClassMapping(numberLabelMap)
        self.hasClassInfo = True

    def setClassInfo(self, classInfoDf: pd.DataFrame):
        if "numeric_label" in classInfoDf:
            classInfoDf = classInfoDf.set_index("numeric_label")
        self.classInfo = classInfoDf
        self.hasClassInfo = True

    @staticmethod
    def _createOutputClassMapping(outputClasses: pd.Series):
        """
        Resolves potential aliases in the output class mapping for unambiguous mask value to class number matching
        """
        classes, numericLabels = np.unique(
            outputClasses.to_numpy(str), return_inverse=True
        )
        outputDf = pd.DataFrame()
        outputDf["label"] = outputClasses
        outputDf["numeric_class"] = numericLabels + 1
        # Add in background for easy indexing
        outputDf.index.name = "numeric_label"
        outputDf.loc[0, ["numeric_class", "label"]] = (0, "BGND")
        # Type not preserved when adding new row for some reason?
        outputDf["numeric_class"] = outputDf["numeric_class"].astype(int)
        return outputDf

    @property
    def numClasses(self):
        return self.classInfo["numeric_class"].max() + 1 if self.hasClassInfo else None

    def generateColoredMask(
        self,
        labelMask: np.ndarray | FilePath,
        outputFile,
        numClasses=None,
        colorMap: str = None,
        resolve=True,
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
            gutils.cvImsaveRgb(outputFile, labelMask)
            return
        if colorMap == "binary":
            # No need to use colormap -- just force high values on save
            # Values > 1 should all clip to 255
            gutils.cvImsaveRgb(outputFile, (labelMask > 0).astype("uint8") * 255)
            return
        if numClasses is None:
            numClasses = np.max(labelMask) + 1
        allColors = fns.getAnyPgColormap(colorMap).getLookupTable(nPts=numClasses)
        item = pg.ImageItem(labelMask, levels=[0, numClasses])
        item.setLookupTable(allColors, update=True)
        item.save(str(outputFile))

    def getMaybeResolve(self, mask: FilePath | np.ndarray, resolve=True):
        if isinstance(mask, FilePath.__args__):
            mask = gutils.cvImreadRgb(self.masksDir / mask, cv.IMREAD_UNCHANGED)
        # Only keep known labels
        if not resolve or not self.hasClassInfo:
            return mask
        mask[
            (mask > 0) & np.isin(mask, self.classInfo.index.to_numpy(), invert=True)
        ] = 0
        # Account for aliasing
        mask[mask > 0] = self.classInfo.loc[mask[mask > 0], "numeric_class"].to_numpy(
            int
        )
        return mask


# Will be the same on a platform so can be cached
@functools.lru_cache()
def getLinkFunc():
    """
    Symlinks rarely have permission by default on windows so be able to copy if needed
    """
    # Use symlinks to avoid lots of file duplication
    def relativeSymlink(src, dst):
        return os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)

    try:
        linkFunc = relativeSymlink
        with tempfile.TemporaryDirectory() as td:
            src: Path = Path(td) / "test"
            src.touch()
            linkFunc(src, src.with_name("testlink"))
    except (PermissionError, OSError):
        linkFunc = shutil.copy
    return linkFunc


_DISCARDED = object()


def stringifyDict(item, unconvertable=(pd.DataFrame, pd.Series)):
    discards = []
    if isinstance(item, dict):
        for kk, vv in item.items():
            vv = stringifyDict(vv)
            item[kk] = vv
            if vv is _DISCARDED:
                discards.append(kk)
    elif isinstance(item, tuple):
        item = tuple(stringifyDict(list(item)))
    elif isinstance(item, list):
        for ii, el in enumerate(item):
            el = stringifyDict(el)
            item[ii] = el
            if el is _DISCARDED:
                discards.append(ii)
    # Special known case which shouldn't be preserved
    elif isinstance(item, unconvertable):
        item = _DISCARDED
    elif not isinstance(item, (int, float, bool, str, type(None))):
        item = str(item)
    # Only happens when item is dict or list
    if discards:
        # Reverse for list case
        for kk in reversed(discards):
            del item[kk]
    return item


def columnsAsPrjParams(df, assignToDf=False):
    cols = list(df.columns)
    for ii, col in enumerate(cols):
        if not isinstance(col, PrjParam):
            cols[ii] = PrjParam(col, type(col)())
    if assignToDf:
        df.columns = cols
        return df
    return cols

class DirCreator(WorkflowDir):
    """
    Convenience class to allow creating workflow dirs from a config file
    """

    def __init__(self, **kwargs):
        kwargs.update(folder=".", createDirs=False, name="Create Directories")
        super().__init__(**kwargs)

    def runWorkflow(self, **kwargs):
        self.parent.createDirs()


def pathCtor(constructor, node):
    return Path(constructor.construct_scalar(node))


fns.loader.constructor.add_constructor("!Path", pathCtor)


class WorkflowEditor(AlgParamEditor):
    def _resolveProccessor(self, proc):
        retProc = super()._resolveProccessor(proc)
        if isinstance(retProc, Workflow_T):
            # Only one top processor can exist, so setting the workflow dir on subfolders
            # will keep each primitive proc at the saveDir level
            retProc.localFolder = self.saveDir
        return retProc
