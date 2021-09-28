from __future__ import annotations

import os.path
import re
import shutil
import typing as t
from pathlib import Path

import pandas as pd

from s3a.compio.exporters import SerialExporter
from s3a.compio.importers import SerialImporter
from s3a import REQD_TBL_FIELDS as RTF
from utilitys import fns
from utilitys.typeoverloads import FilePath

def default_title(name, trim_exprs, prefix, suffix):
    for suff in trim_exprs:
        name = re.sub(f'_?{suff}', '', name)
    name = name + suffix
    if isinstance(prefix, RegisteredPath):
        name = os.path.join(prefix.sub_path, name)
    elif prefix:
        name = prefix + name
    return name

def title_case(*args):
    name = default_title(*args)
    return fns.pascalCaseToTitle(name)

class RegisteredPath:

    def __init__(self,
                 suffix='',
                 prefix: str | RegisteredPath=None,
                 trim_exprs=('file', 'path', 'dir'),
                 title: t.Callable=default_title,
                 output=True
                 ):
        self.prefix = prefix
        self.suffix = suffix
        self.trim_exprs = trim_exprs
        self.output = output
        self.title = title

    def __set_name__(self, owner, name):
        # See utilitys.misc.DeferredActionStackMixin for description of why copy() is used
        name = self.title(name, self.trim_exprs, self.prefix, self.suffix)
        self.sub_path = name
        if self.output:
            paths = owner.output_paths.copy()
            paths.add(name)
            owner.output_paths = paths

    def __get__(self, obj, objtype):
        if isinstance(obj, type):
            return self
        ret = obj.workflow_dir / self.sub_path
        return ret


class WorkflowDir:
    output_paths: t.Set[str] = set()

    name: str = None

    def __init__(
        self,
        folder: Path | str,
        *,
        config: dict=None,
        reset=False,
        create_dirs=False
    ):
        self.workflow_dir = Path(folder)
        self.config = config

        if self.name is None:
            self.name = type(self).__name__.replace('Workflow', '').lower()

        if reset:
            self.reset()
        if create_dirs:
            self.create_dirs()
        if config is not None:
            fns.saveToFile(config, self.workflow_dir/(self.name + '_config.yml'))

    def reset(self):
        # Sort so parent paths are deleted first during rmtree
        for path in sorted(self.output_paths):
            path = self.workflow_dir / path
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def create_dirs(self, exclude_exprs=('.',)):
        # Sort so parent paths are created first
        for path in sorted(self.output_paths):
            if any(expr in path for expr in exclude_exprs):
                continue
            self.workflow_dir.joinpath(path).mkdir(exist_ok=True)

class S3AFeatureWorkflow(WorkflowDir):
    formatted_input_path = RegisteredPath()

    _converted_inputs_dir = None
    """
    If set, this directory can be used to check against which formatted inputs have already been processed
    """

    def create_formatted_inputs(self, annotation_path: FilePath=None):
        """
        Generates cleansed csv files from the raw input dataframe. Afterwards, saves annotations in files separated
        by image to allow multiprocessing on subsections of components
        """
        if annotation_path is None:
            return pd.DataFrame()
        if annotation_path.is_dir():
            df = fns.readDataFrameFiles(annotation_path, SerialImporter.readFile)
        else:
            df = SerialImporter.readFile(annotation_path)
        for image, subdf in df.groupby(RTF.IMG_FILE.name):
            newName = Path(image).with_suffix('.csv').name
            dest = self.formatted_input_path/newName
            if not dest.exists():
                SerialExporter.writeFile(dest, subdf, readonly=False)
        return df

    @property
    def new_input_files(self):
        """
        Helper property to act like a list of files from the cleansed inputs
        """
        files = self.formatted_input_path.glob('*.csv')

        if self._converted_inputs_dir is None:
            return fns.naturalSorted(files)

        generated = {f.stem for f in self._converted_inputs_dir.glob('*.*')}
        return fns.naturalSorted(f for f in files if f.stem not in generated)
