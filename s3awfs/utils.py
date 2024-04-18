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
from typing import List
import warnings
import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, interact
from qtextras import OptionsDict, fns
from qtextras.typeoverloads import FilePath
from s3a import generalutils as gutils
from s3a.parameditors.algcollection import AlgorithmEditor, PipelineFunction
from s3a.processing import PipelineParameter, maybe_get_function

T = t.TypeVar("T")
_ParentType = t.Callable[[], "NestedWorkflow"]


def get_stage(parameter: Parameter) -> PipelineParameter | PipelineFunction | None:
    """If parameter is a PipelineParameter, returns that. Else, returns the PipelineFunction
    if it has one, else None.

    Args:
        parameter (Parameter): parameter

    Returns:
        PipelineParameter | PipelineFunction | None: An oddly parametric return
    """
    if isinstance(parameter, PipelineParameter):
        return parameter
    return maybe_get_function(parameter)


def default_title(
    name: str, trim_exprs: List[str], prefix: RegisteredPath | str | None, suffix: str
) -> Path | str:
    """Turns the given name into a title with the given customization options.

    Args:
        name (str): Pascal case title
        trim_exprs (List[str]): Expressions to remove from the name
        prefix (RegisteredPath | str | None): Prefix to prepend to the name
        suffix (str): Suffix to add to the name

    Returns:
        Path | str: Title formatted name
    """
    name = fns.pascalCaseToTitle(name).replace(" ", "_").lower() + suffix
    for suff in trim_exprs:
        name = re.sub(f"_?{suff}", "", name)
    if isinstance(prefix, RegisteredPath):
        name = os.path.join(prefix.subPath, name)
    elif prefix:
        name = prefix + name
    return name


def title_case(*args) -> Path | str:
    """Passthrough to default_title.

    Returns:
        Path | str: the return from default_title
    """
    warnings.warn(
        "The title_case function is deprecated", DeprecationWarning, stacklevel=2
    )
    return default_title(*args)


# Credit: https://stackoverflow.com/a/37095733/9463643
def path_is_parent(parent_path, child_path):
    # Smooth out relative path names, note: if you are concerned about symbolic links,
    # you should use os.path.realpath too
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    # Compare the common path of the parent and child path with the common path of just
    # the parent path. Using the commonpath method on just the parent path will
    # regularise the path name in the same way as the comparison that deals with both
    # paths, removing any trailing path separator
    return parent_path == os.path.commonpath([parent_path, child_path])


class RegisteredPath:
    def __init__(
        self,
        suffix="",
        prefix: str | RegisteredPath | None = None,
        trimExprs=("file", "path", "dir"),
        title: t.Callable = default_title,
        output=True,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.trimExprs = trimExprs
        self.output = output
        self.title = title

    def __set_name__(self, owner, name):
        # See qtextras.misc.DeferredActionStackMixin for description of why copy() is used
        name = self.title(name, self.trimExprs, self.prefix, self.suffix)
        self.subPath = name
        if self.output:
            paths = owner.outputPaths.copy()
            paths.add(name)
            owner.outputPaths = paths

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return obj.workflowPath / self.subPath

    def __fspath__(self):
        return self.subPath


class WorkflowMixin:
    name: str | None = None
    parent: _ParentType

    def __init__(self, folder: FilePath | Path | str | None = None):
        self.localFolder = Path(folder or "")

    @property
    def workflowPath(self):
        if (parent := self.parent()) and hasattr(parent, "workflowPath"):
            return parent.workflowPath / self.localFolder
        return self.localFolder

    def resetRegisteredPaths(self): ...

    def createDirectories(self): ...


class WorkflowDirectory(PipelineFunction, WorkflowMixin):
    outputPaths: t.Set[str] = set()
    parent: _ParentType

    def __init__(
        self,
        name: str | None = None,
        folder: Path | str | None = None,
        *,
        reset=False,
        createDirectories=False,
        **kwargs,
    ):
        baseName = self.name or type(self).__name__.replace("Workflow", "")
        defaultName = fns.pascalCaseToTitle(baseName)
        name = name or defaultName

        if folder is None:
            folder = name

        PipelineFunction.__init__(self, self.runWorkflow, name=name, **kwargs)
        WorkflowMixin.__init__(self, folder)

        if reset:
            self.resetRegisteredPaths()
        if createDirectories:
            self.createDirectories()

    def resetRegisteredPaths(self):
        # Sort so parent paths are deleted first during rmtree
        for path in sorted(self.outputPaths):
            path = self.workflowPath / path
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def createDirectories(self, excludeExprs=(".",)):
        # Sort so parent paths are created first
        self.workflowPath.mkdir(exist_ok=True)
        for path in sorted(self.outputPaths):
            if any(expr in path for expr in excludeExprs):
                continue
            self.workflowPath.joinpath(path).mkdir(exist_ok=True)

    def runWorkflow(self, *args, **kwargs):
        pass


class NestedWorkflow(PipelineParameter, WorkflowMixin):
    stages: list[WorkflowDirectory]

    def _stageSummaryWidget(self):
        pass

    def __init__(
        self,
        name: str | None = None,
        folder: FilePath | str | None = None,
        createDirectories=False,
        reset=False,
    ):
        PipelineParameter.__init__(self, name=name or "<Unnamed>")
        WorkflowMixin.__init__(self, folder or name)

        if reset:
            self.resetRegisteredPaths()
        if createDirectories:
            self.createDirectories()

        self.setOpts(folder=folder)

    def disableStages(self, *stageClasses: t.Type[WorkflowDirectory]):
        for cls in stageClasses:
            self.get(cls).disabled = True

    def saveStringifiedConfig(self, folder: str | Path = None, **initKwargs):
        state = self.saveState(filter=("defaults", "meta"))

        # Some values are unrepresentable in their natural form (e.g. Paths)
        state = stringifyDict(state)
        if folder is None:
            folder = self.workflowPath
        fns.saveToFile(state, Path(folder).joinpath("config.yml"))
        return state

    def getTitleAndMaybeMetadata(self, includeMeta=True):
        title, meta = super().getTitleAndMaybeMetadata()
        if not includeMeta:
            return title, meta
        meta["folder"] = self.localFolder
        return title, meta

    @classmethod
    def splitInitAndRunKwargs(cls, kwargs):
        """
        Converts a dict of potentially both __init__ keywords and run() keywords into
        two separate dicts
        """
        initSpec = set(inspect.signature(cls.__init__).parameters)
        initKwargs = {}
        for kw in initSpec:
            if kw in kwargs:
                initKwargs[kw] = kwargs.pop(kw)
        return initKwargs, kwargs

    def addStage(self, stage, *, cache=False, **kwargs):
        # Changes the 'cache' default to False instead of True in base class
        return super().addStage(stage, cache=cache, **kwargs)

    def addWorkflow(self, workflowClass: t.Type[T], **kwargs) -> T:
        wf = workflowClass(**kwargs)
        return self.addStage(wf)

    def resetRegisteredPaths(self):
        for wf in self:
            if wf.opts["enabled"]:
                wf.resetRegisteredPaths()

    def createDirectories(self, excludeExprs=(".",)):
        self.workflowPath.mkdir(exist_ok=True)
        for wf in self:
            if not wf.opts["enabled"]:
                continue
            if func := maybe_get_function(wf):
                wf = func
            if isinstance(wf, Workflow_T):
                wf.createDirectories(excludeExprs)

    def _getFromRoot(self, root: Workflow_T, wfClassorName):
        # TODO: Prefer stages closer to requesting process
        if (
            not isinstance(wfClassorName, str) and isinstance(root, wfClassorName)
        ) or root.title() == wfClassorName:
            return root
        if not isinstance(root, PipelineParameter):
            return None
        for child in root:
            if ret := self._getFromRoot(get_stage(child), wfClassorName):
                return ret
        # No matches anywhere in the tree
        return None

    def get(self, wfClass: t.Type[T] | str, addIfMissing=True) -> T:
        stage = get_stage(self)
        while (parent := stage.parent()) and get_stage(parent):
            stage = get_stage(parent)
        if match := self._getFromRoot(stage, wfClass):
            return match
        # Stage not present already; add at self's level
        if addIfMissing and not isinstance(wfClass, str):
            stage = self.addWorkflow(wfClass)
            stage.setOpts(enabled=False)
            return stage
        # Requested stage type is not present
        raise KeyError(f'Workflow type "{wfClass}" is not present')


Workflow_T = (WorkflowDirectory, NestedWorkflow)


def argparseHelpAction(nested: NestedWorkflow):
    class NestedWorkflowHelp(argparse.Action):
        def __init__(self, **kwargs):
            kwargs.update(nargs=0)
            super().__init__(**kwargs)

        def __call__(self, parser, *args, **kwargs) -> None:
            state = {}
            for stage in nested.flattenedFunctions():
                pgDict = interact.functionToParameterDict(stage.function)
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
        numberLabelMap: pd.Series = None,
        labelMaskSource=None,
    ):
        """
        Parameters
        ----------
        numberLabelMap
            Tracks the relationship between numeric mask values and class labels. Its
            index is the numeric mask value and the value is the class label (can also
            be numeric)
        """
        self.masksDir = labelMaskSource or Path()
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
        self.classInfo = self.createOutputClassMapping(numberLabelMap)
        self.hasClassInfo = True

    def setClassInfo(self, classInfoDf: pd.DataFrame):
        if "numeric_label" in classInfoDf:
            classInfoDf = classInfoDf.set_index("numeric_label")
        self.classInfo = classInfoDf
        self.hasClassInfo = True

    @staticmethod
    def createOutputClassMapping(outputClasses: pd.Series):
        """
        Resolves potential aliases in the output class mapping for unambiguous mask
        value to class number matching
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

        Parameters
        ----------
        labelMask
            numpy image to transform (or input file containing a mask)
        outputFile
            Location to export the transformed image
        colorMap
            A string of the Matplotlib color map to use for the generated RGB ground
            truth segmentation masks. Acceptable color maps are restrained to the
            following: https://matplotlib.org/stable/tutorials/colors/colormaps.html.
            If *None*, uses the raw label mask without any changes. If "binary",
            turns any pixel > 0 to white and any pixel = 0 as black
        numClasses
            Total number of classes for scaling rgb values. If *None*, uses the number
            of classes present in the image. Can also be set to ``self.numClasses`` for
            global reference.
        resolve
            Whether the mask should be resolved (i.e. does it contain numeric aliases?)
            or not (i.e. was it already resolved from a previous call to ``self.get()``?)
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


def columnsAsOptions(df, assignToDf=False):
    cols = list(df.columns)
    for ii, col in enumerate(cols):
        if not isinstance(col, OptionsDict):
            cols[ii] = OptionsDict(col, type(col)())
    if assignToDf:
        df.columns = cols
        return df
    return cols


class CreateDirectories(WorkflowDirectory):
    """
    Convenience class to allow creating workflow dirs from a config file
    """

    def __init__(self, **kwargs):
        kwargs.update(folder=".", createDirectories=False)
        super().__init__(**kwargs)

    def runWorkflow(self, **kwargs):
        self.parent().createDirectories()


def pathCtor(constructor, node):
    return Path(constructor.construct_scalar(node))


fns.loader.constructor.add_constructor("!Path", pathCtor)


class WorkflowEditor(AlgorithmEditor):
    def _resolveProccessor(self, proc):
        retProc = super()._resolveProccessor(proc)
        if isinstance(retProc, Workflow_T):
            # Only one top processor can exist, so setting the workflow dir on subfolders
            # will keep each primitive proc at the directory level
            retProc.localFolder = self.directory
        return retProc
