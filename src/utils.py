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

class SnapTargetItem(pg.TargetItem):

    def __init__(self, snapItem, *args, **kwargs):
        self.snapItem = snapItem
        super().__init__(*args, **kwargs)
        self.snapIndex = 0

    def setPos(self, pos):
        curve_data = np.c_[self.snapItem.getData()]
        closest_pt_idx = np.argmin(np.linalg.norm(curve_data - [pos.x(), pos.y()], axis=1))
        closest_pt = QtCore.QPointF(*curve_data[closest_pt_idx])
        self.snapIndex = closest_pt_idx
        return super().setPos(closest_pt)

class NComponentVisualizer(widgets.ImageViewer):

    def __init__(self, samples_df: pd.DataFrame, model=None, **kwargs):
        samples_df = samples_df.reset_index(drop=True)
        # Keep reference to colorbar plot if shown else garbage collection will occur
        self.active_colorbar = None


        if model is None:
            model = PCA()
            # Turn MxNxChan images into 1x(M*N*Chan) feature vectors
            features = np.vstack(samples_df['image'].apply(np.ndarray.ravel))
            model.fit(features)

        # Avoid too much representative data by just keeping one sample from each label
        samples_df = samples_df.groupby('label').apply(lambda el: el.sample(n=1, random_state=42)).reset_index(drop=True)

        super().__init__(samples_df.at[0, 'image'], **kwargs)
        self.samples_df = samples_df
        self.image_shape = samples_df.iat[0, samples_df.columns.get_loc('image')].shape
        self.model = model
        self.statMsg = QtWidgets.QLabel()
        self.props = ParamContainer()

        self.variance_plot = pg.PlotWidget()
        self.variance_plot.setMaximumHeight(200)
        self.tot_variance = np.cumsum(model.explained_variance_ratio_)
        curveItem = pg.PlotCurveItem(self.tot_variance, pen=pg.mkPen(width=3))
        self.variance_plot.addItem(curveItem)
        self.marker = SnapTargetItem(curveItem, pos=pg.Point([0, self.tot_variance[0]]),
                                     labelOpts=dict(anchor=(0.5, 0)), label=True)
        self.variance_plot.addItem(self.marker)

        NComponentVisualizer.shouldXform = self.toolsEditor.registerProp(
            PrjParam('Inverse Transform', True), container=self.props)

        self.toolsEditor.registerFunc(
            self.show_original_overlay,
            runOpts=RunOpts.ON_CHANGED,
            nest=False,
            container=self.props
        )

        def on_change(_):
            self.props['n_components'] = self.marker.snapIndex
            if self.marker.label():
                self.marker.label().setText(f'Tot. Variance: {self.tot_variance[self.marker.snapIndex]:0.2f}\n'
                                            f'# Components: {self.marker.snapIndex+1}')

        self.marker.sigPositionChanged.connect(
            on_change
        )

        self.toolsEditor.registerFunc(
            self.change_sample,
            runOpts=RunOpts.ON_CHANGED,
            container=self.props
        )
        self.toolsEditor.registerFunc(
            self.try_n_components,
            runOpts=RunOpts.ON_CHANGING,
            container=self.props
        )
        self.toolsEditor.registerFunc(
            self.plot_components,
        )
        self.toolsEditor.registerFunc(
            self.show_colorbar
        )


        def spawn():
            widgets.safeSpawnDevConsole(self)
        self.toolsEditor.registerFunc(spawn, name='Dev Console')

        limits = {kk: vv for kk, vv in zip(samples_df['label'], samples_df.index)}
        self.props.params['sample_index'].setOpts(limits=limits)
        # self.props.params['n_components'].setOpts(limits=[0, model.n_components_-1])
        on_change(None)

        def update_ncomp_param(p, val):
            self.marker.setPos(pg.Point(val, self.tot_variance[val]))
        self.props.params['n_components'].sigValueChanged.connect(update_ncomp_param)

    def _widgetContainerChildren(self):
        return [self, self.variance_plot]


    def change_sample(self, sample_index=0):
        """
        :param sample_index:
        type: list
        """
        self.statMsg.setText(f'Label: {self.samples_df.at[sample_index, "label"]}')
        self.try_n_components(self.props['n_components'])

    def try_n_components(self, n_components=1):
        if self.props['show_original']:
            return
        model = self.model
        samples_df = self.samples_df
        if self.shouldXform:
            xformed = model.transform(samples_df.at[self.props['sample_index'], 'image'].reshape(1,-1))
            inverse = xformed[:, :n_components] @ model.components_[:n_components] + model.mean_
        else:
            # Image raw PCA scores
            inverse = np.abs(model.components_[n_components])
        self.setImage(inverse.reshape(*self.image_shape))

    def show_original_overlay(self, show_original=False):
        if show_original:
            self.setImage(self.samples_df.at[self.props['sample_index'], 'image'])
        else:
            self.try_n_components(self.props['n_components'])

    def plot_components(self, n_components=9):
        infos = []
        for ii, component in enumerate(self.model.components_[:n_components]):
            infos.append(dict(
                image=np.abs(component.reshape(*self.image_shape)),
                name=f'Component {ii+1}'
            ))
        oldOpts = dict(foreground=pg.getConfigOption('foreground'), background=pg.getConfigOption('background'))
        try:
            pg.setConfigOptions(foreground='k', background='w')
            dummy = ImageProcess()
            dummy.getAllStageInfos = lambda *args, **kwds: infos
            dummy.stageSummary_gui()
        finally:
            pg.setConfigOptions(**oldOpts)


    def show_colorbar(self):
        shape = self.image_shape
        if len(shape) == 2:
            shape = (*shape, 1)
        nchans = shape[-1]
        bar = []
        for comp in self.model.components_:
            mean = np.mean(np.abs(comp.reshape(-1, nchans)), axis=0)
            bar.append(mean/np.max(mean))
        bar = np.vstack(bar)
        bar *= self.model.explained_variance_ratio_[:, None]
        bar /= bar.max()
        width = np.prod(shape)
        plt.imshow(bar.reshape(1, -1, nchans), extent=[0, width, 0, 0.05*width])
        plt.gca().get_yaxis().set_visible(False)
        plt.title('Mean color of absolute PCA weights, with variance weighting')
        plt.xlabel('Component #')
        plt.show()



if __name__ == '__main__':
    samples_dict = {
        'image': [np.random.random((50, 50, 3))]*10,
        'label': list('abcdefghij')
    }
    pg.mkQApp()
    samples_frame = pd.DataFrame(samples_dict)
    viz = NComponentVisualizer(samples_frame)
    viz.show_exec()
