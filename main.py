import sys
import typing as t
from pathlib import Path

import pandas as pd
# Some systems need qt initialized before cv gets imported
import pyqtgraph as pg

pg.mkQApp()

from utilitys import fns
from workflows.constants import SMD_INIT_OPTS, DEFAULT_RESIZE_OPTS, FPIC_SMDS
from workflows.tvtsplit import TrainValidateTestSplitWorkflow
from workflows.png import PngExportWorkflow
from workflows.compimgs import ComponentImagesWorkflow
from workflows.regionpropfeats import RegionPropertiesWorkflow
from workflows.utils import NestedWorkflow, WorkflowDir

class MainWorkflow(NestedWorkflow):
    name = 'Main Workflow'
    def __init__(
        self,
        folder,
        trainLinknet=True,
        **kwargs
    ):
        name = kwargs.pop('name', self.name)
        super().__init__(folder, name=name, **kwargs)
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

def main(outputFolder, labelMapFile, inputAnnPath='', trainLinknet=True, reset=False, **kwargs):
    mwf = MainWorkflow(
        outputFolder,
        createDirs=True,
        trainLinknet=trainLinknet,
        reset=reset
    )
    # mwf.disableStages(ComponentImagesWorkflow, PngExportWorkflow, RegionPropertiesWorkflow, TrainValidateTestSplitWorkflow)
    mwf.run(
        annotationPath=inputAnnPath or FPIC_SMDS,
        labelMap=pd.read_csv(labelMapFile, index_col=['numeric_label']),
        **SMD_INIT_OPTS,
        resizeOpts=DEFAULT_RESIZE_OPTS,
        **kwargs
    )

    # Some values are unrepresentable in their natural form (e.g. Paths)
    def cvt(item):
        if isinstance(item, dict):
            for kk, vv in item.items():
                item[kk] = cvt(vv)
        elif isinstance(item, tuple):
            item = tuple(cvt(list(item)))
        elif isinstance(item, list):
            for ii, el in enumerate(item):
                item[ii] = cvt(el)
        elif not isinstance(item, (int, float, bool, str, type(None))):
            item = str(item)
        return item

    state = cvt(mwf.saveState(includeDefaults=True))
    fns.saveToFile(state, mwf.workflowDir / 'config.yml')

def main_rgbFeatures512(**kwargs):
    outFolder = kwargs.pop('outputFolder', None) or Path.home()/'Desktop/rgb_features_512'
    labelMapFile = kwargs.pop('labelMapFile', None) or outFolder/'aliased_labels.csv'
    main(**kwargs, outputFolder=outFolder, labelMapFile=labelMapFile)

def main_cli():
    parser = fns.makeCli(main)
    # Prevent failure for rgb512 parsing

    argv = sys.argv[1:].copy()
    for kk in 'outputFolder', 'labelMapFile':
        if f'--{kk}' not in argv:
            argv += [f'--{kk}', None]
    args = parser.parse_args(argv)
    main_rgbFeatures512(**vars(args))

if __name__ == '__main__':
    main_cli()
