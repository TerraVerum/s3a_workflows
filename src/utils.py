from __future__ import annotations

import re
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from s3a.processing import ImageProcess
from sklearn.decomposition import PCA
from utilitys import widgets, RunOpts, ParamContainer, PrjParam

class RegisteredPath:

    def __init__(self, extra='', trim_exprs=('file', 'path')):
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
        ret = obj.folder/self.sub_path
        return ret

class ExportDir:
    registered_paths: t.Set[str] = set()

    def __init__(self, folder: Path | str):
        self.folder = Path(folder)

    def reset(self):
        for path in self.registered_paths:
            path = self.folder/path
            if not path.exists():
                continue
            if path.is_dir():
                for file in path.iterdir():
                    file.unlink()
            else:
                path.unlink()

    def make_dirs(self, exclude_exprs=('.',)):
        for path in self.registered_paths:
            if any(expr in path for expr in exclude_exprs):
                continue
            self.folder.joinpath(path).mkdir(exist_ok=True)