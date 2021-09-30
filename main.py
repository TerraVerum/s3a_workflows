import typing as t
from pathlib import Path
import datetime

import pandas as pd
# Some systems need qt initialized before cv gets imported
import pyqtgraph as pg

pg.mkQApp()

from workflows.constants import FPIC_IMAGES, FPIC_SMDS, SMD_INIT_OPTS, DEFAULT_RESIZE_OPTS
from workflows.imagefeats import ComponentImagesWorkflow, PngExportWorkflow, TrainValTestWorkflow
from workflows.regionpropfeats import RegionPropertiesWorkflow
from workflows.utils import WorkflowDir, NestedWorkflow


class MainWorkflow(NestedWorkflow):
    def __init__(
        self,
        folder,
        trainLinknet=True,
        **kwargs
    ):
        super().__init__(folder, **kwargs)
        # Just to make the function name shorter
        _ = self.addWorkflow

        self.imgWf        = _(ComponentImagesWorkflow, **kwargs)
        self.regionpropWf = _(RegionPropertiesWorkflow, **kwargs)
        self.imgExportWf = _(PngExportWorkflow, **kwargs)
        self.tvtWf        = _(TrainValTestWorkflow, **kwargs)
        if trainLinknet:
            self.linkWf   = self.createAddLinknet(**kwargs)
        else:
            self.linkWf   = None


    def createAddLinknet(self, **kwargs):
        # Defer to avoid tensorflow gpu initialization where possible
        from workflows.models.linknet import LinkNetTrainingWorkflow
        return self.addWorkflow(LinkNetTrainingWorkflow, **kwargs)

if __name__ == '__main__':
    mwfFolder = Path.home()/'Desktop/rgb_features_512'
    mwf = MainWorkflow(
        mwfFolder,
        createDirs=True,
        trainLinknet=False,
        # resetRegisteredPaths=True
    )
    # for wfClass in ComponentImagesWorkflow, PngExportWorkflow, RegionPropertiesWorkflow:
    #     mwf.get(wfClass).disabled = True
    mwf.run(
        # annotationPath=mwf.workflowDir/'subset',
        labelMap=pd.read_csv(mwf.workflowDir / 'aliased_labels.csv', index_col=['numeric_label']),
        testOnUnused=True,
        maxTestSamps=100,
        **SMD_INIT_OPTS,
        resizeOpts=DEFAULT_RESIZE_OPTS
    )
