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
        if obj is None:
            return self
        return obj.workflow_dir / self.sub_path

    def __fspath__(self):
        return self.sub_path

# Simple descriptor since @property doesn't work for classes
class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class WorkflowDir:
    output_paths: t.Set[str] = set()
    _name: str = None
    default_config: dict[str, t.Any] = {}

    def __init__(
        self,
        folder: Path | str,
        *,
        config: dict=None,
        reset=False,
        create_dirs=False,
        **kwargs
    ):
        self.workflow_dir = Path(folder)
        self.config = self.default_config.copy()
        if config:
            fns.hierarchicalUpdate(self.config, config, replaceLists=True)

        if reset:
            self.reset()
        if create_dirs:
            self.create_dirs()
        if self.config != self.default_config:
            fns.saveToFile(config, self.workflow_dir/(self.name + '_config.yml'))

    # It's a class-level property, that's why this is false positive
    # noinspection PyMethodParameters
    @classproperty
    def name(cls: t.Type[WorkflowDir]):
        return cls._name or fns.pascalCaseToTitle(cls.__name__.replace('Workflow', ''))

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
        self.workflow_dir.mkdir(exist_ok=True)
        for path in sorted(self.output_paths):
            if any(expr in path for expr in exclude_exprs):
                continue
            self.workflow_dir.joinpath(path).mkdir(exist_ok=True)

    def run(self, *args, **kwargs):
        pass

class AliasedMaskResolver:
    class_info: pd.DataFrame

    def __init__(
        self,
        number_label_map=None,
        label_masks_dir=None,
    ):
        """
        :param number_label_map: pd.Series
            Tracks the relationship between numeric mask values and class labels. Its index is the numeric mask value
            and the value is the class label (can also be numeric)
        """
        self.masks_dir = label_masks_dir or Path()

        self.has_class_info = number_label_map is not None
        if not self.has_class_info:
            # Accept all labels as unique
            number_label_map = np.arange(np.iinfo('uint16').max)
        self.set_number_label_map(number_label_map)

    def set_number_label_map(self, number_label_map):
        if not isinstance(number_label_map, pd.Series):
            number_label_map = pd.Series(number_label_map, number_label_map)
        self.class_info = self._create_output_class_mapping(number_label_map)
        self.has_class_info = True

    def set_class_info(self, class_info_df: pd.DataFrame):
        if 'numeric_label' in class_info_df:
            class_info_df = class_info_df.set_index('numeric_label')
        self.class_info = class_info_df
        self.has_class_info = True

    @staticmethod
    def _create_output_class_mapping(output_classes: pd.Series):
        """
        Resolves potential aliases in the output class mapping for unambiguous mask value to class number matching
        """
        classes, numeric_labels = np.unique(output_classes.to_numpy(str), return_inverse=True)
        output_df = pd.DataFrame()
        output_df['label'] = output_classes
        output_df['numeric_class'] = numeric_labels + 1
        # Add in background for easy indexing
        output_df.index.name = 'numeric_label'
        output_df.loc[0, ['numeric_class', 'label']] = (0, 'BGND')
        # Type not preserved when adding new row for some reason?
        output_df['numeric_class'] = output_df['numeric_class'].astype(int)
        return output_df

    @property
    def num_classes(self):
        return len(self.class_info['numeric_class'].unique()) if self.has_class_info else None

    def generate_colored_mask(
        self,
        label_mask: np.ndarray | FilePath,
        output_file,
        num_classes=None,
        color_map: str=None,
        resolve=True
    ):
        """
        A helper function that generates rescaled or RGB versions of label masks
        :param label_mask: numpy image to transform (or input file containing a mask)
        :param output_file: Location to export the transformed image
        :param color_map: A string of the Matplotlib color map to use for the generated RGB ground truth segmentation masks.
            Acceptable color maps are restrained to the following:
             https://matplotlib.org/stable/tutorials/colors/colormaps.html. If *None*, uses the raw label mask without
             any changes. If "binary", turns any pixel > 0 to white and any pixel = 0 as black
        :param num_classes: Total number of classes for scaling rgb values. If *None*, uses the number of classes
            present in the image. Can also be set to ``self.num_classes`` for global reference.
        :param resolve: Whether the mask should be resolved (i.e. does it contain numeric aliases?) or not (i.e.
          was it already resolved from a previous call to ``self.get()``?)
        """
        label_mask = self.get_maybe_resolve(label_mask, resolve=resolve)
        if color_map is None:
            gutils.cvImsave_rgb(output_file, label_mask)
            return
        if color_map == 'binary':
            # No need to use colormap -- just force high values on save
            # Values > 1 should all clip to 255
            gutils.cvImsave_rgb(output_file, (label_mask > 0).astype('uint8') * 255)
            return
        if num_classes is None:
            num_classes = np.max(label_mask)
        all_colors = pg.colormap.get(color_map).getLookupTable(nPts=num_classes)
        item = pg.ImageItem(label_mask, levels=[0, num_classes])
        item.setLookupTable(all_colors, update=True)
        item.save(str(output_file))

    def get_maybe_resolve(self, mask: FilePath | np.ndarray, resolve=True):
        if isinstance(mask, FilePath.__args__):
            mask = gutils.cvImread_rgb(self.masks_dir/mask, cv.IMREAD_UNCHANGED)
        # Only keep known labels
        if not resolve:
            return mask
        mask[(mask > 0) & ~np.isin(mask, self.class_info.index)] = 0
        # Account for aliasing
        # Ravel since pandas doesn't like 2d indexing to broadcast
        mask[mask > 0] = self.class_info.loc[mask[mask > 0], 'numeric_class'].to_numpy(int)
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
