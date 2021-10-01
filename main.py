from pathlib import Path
import typing as t

import pandas as pd
# Some systems need qt initialized before cv gets imported
import pyqtgraph as pg

pg.mkQApp()

from workflows.constants import SMD_INIT_OPTS, DEFAULT_RESIZE_OPTS
from workflows.trainvaltest import TrainValidateTestSplitWorkflow
from workflows.png import PngExportWorkflow
from workflows.compimgs import ComponentImagesWorkflow
from workflows.regionpropfeats import RegionPropertiesWorkflow
from workflows.utils import NestedWorkflow, WorkflowDir

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
        self.tvtWf        = _(TrainValidateTestSplitWorkflow, **kwargs)
        if trainLinknet:
            self.linkWf   = self.createAddLinknet(**kwargs)
        else:
            self.linkWf   = None

    def createAddLinknet(self, **kwargs):
        # Defer to avoid tensorflow gpu initialization where possible
        from workflows.models.linknet import LinkNetTrainingWorkflow
        return self.addWorkflow(LinkNetTrainingWorkflow, **kwargs)

    def disableStages(self, *stageClasses: t.Type[WorkflowDir]):
        for cls in stageClasses:
            self.get(cls).disabled = True

if __name__ == '__main__':
    mwfFolder = Path.home()/'Desktop/rgb_features_512'
    mwf = MainWorkflow(
        mwfFolder,
        createDirs=True,
        trainLinknet=False,
        # resetRegisteredPaths=True
    )
    # mwf.disableStages(ComponentImagesWorkflow, PngExportWorkflow, RegionPropertiesWorkflow, TrainValidateTestSplitWorkflow)
    mwf.run(
        annotationPath=mwf.workflowDir/'subset',
        labelMap=pd.read_csv(mwf.workflowDir / 'aliased_labels.csv', index_col=['numeric_label']),
        maxTestSamps=100,
        **SMD_INIT_OPTS,
        resizeOpts=DEFAULT_RESIZE_OPTS,
    )
