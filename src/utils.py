from __future__ import annotations

import re
import typing as t
from pathlib import Path

from s3a.compio.exporters import SerialExporter
from s3a.compio.importers import SerialImporter
from s3a import REQD_TBL_FIELDS as RTF
from utilitys import fns
from utilitys.typeoverloads import FilePath

class RegisteredPath:

    def __init__(self, extra='', trim_exprs=('file', 'path', 'dir')):
        self.extra = extra
        self.trim_exprs = trim_exprs

    def __set_name__(self, owner, name):
        for suff in self.trim_exprs:
            name = re.sub(f'_?{suff}', '', name)
        if self.extra:
            name = name + self.extra
        self.sub_path = name
        owner.registered_paths.add(name)

    def __get__(self, obj, objtype):
        if isinstance(obj, type):
            return self
        ret = obj.workflow_folder / self.sub_path
        return ret

class WorkflowDir:
    registered_paths: t.Set[str] = set()

    def __init__(self, folder: Path | str):
        self.workflow_folder = Path(folder)

    def reset(self):
        for path in self.registered_paths:
            path = self.workflow_folder / path
            if not path.exists():
                continue
            if path.is_dir():
                for file in path.iterdir():
                    file.unlink()
            else:
                path.unlink()

    def create_dirs(self, exclude_exprs=('.',)):
        for path in self.registered_paths:
            if any(expr in path for expr in exclude_exprs):
                continue
            self.workflow_folder.joinpath(path).mkdir(exist_ok=True)

class S3AFeatureWorkflow(WorkflowDir):
    formatted_input_path = RegisteredPath()

    def create_formatted_inputs(self, annotation_path: FilePath):
        """
        Generates cleansed csv files from the raw input dataframe. Afterwards, saves annotations in files separated
        by image to allow multiprocessing on subsections of components
        """
        if annotation_path.is_dir():
            df = fns.readDataFrameFiles(annotation_path, SerialImporter.readFile)
        else:
            df = SerialImporter.readFile(annotation_path)
        for image, subdf in df.groupby(RTF.IMG_FILE.name):
            SerialExporter.writeFile(self.formatted_input_path/(image + '.csv'), subdf, readonly=False)
        return df

    def run(self, annotation_path, reset=False):
        """
        Top-level function. Takes either a csv file or folder of csvs and produces the final result. So, this method
        will show the order in which all processes should be run
        """
        self.create_dirs()
        if reset:
            self.reset()

        self.create_formatted_inputs(annotation_path)
        return self.feature_workflow()

    def feature_workflow(self):
        pass
