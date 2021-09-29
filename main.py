import typing as t
from pathlib import Path

import pandas as pd
# Some systems need qt initialized before cv gets imported
import pyqtgraph as pg
pg.mkQApp()

from workflows.constants import FPIC_IMAGES, FPIC_SMDS
from workflows.features import RegionPropsWorkflow, CompImgsWorkflow
from workflows.features.imagefeats import CompImgsExportWorkflow, TrainValTestWorkflow
from workflows.models.linknet import LinkNetWorkflow
from workflows.utils import WorkflowDir

T = t.TypeVar('T')

class MainWorkflow(WorkflowDir):
    def __init__(self, folder, comp_imgs_config=None, **kwargs):
        folder = Path(folder)
        self.all_workflows = []
        self.img_wf = self.add_workflow(CompImgsWorkflow(folder/'1. Component Image Extraction', comp_imgs_config, **kwargs))
        self.regionprop_wf = self.add_workflow(RegionPropsWorkflow(folder/'2. Region Props Extraction', **kwargs))
        self.img_export_wf = self.add_workflow(CompImgsExportWorkflow(folder/'3. Raw Png Exports', **kwargs))
        self.tvt_wf = self.add_workflow(TrainValTestWorkflow(folder / '4. Label Antialias + Train-Val-Test Split', **kwargs))
        self.link_wf = self.add_workflow(LinkNetWorkflow( folder / '5. LinkNet Model Outputs',  **kwargs))

        super().__init__(folder, **kwargs)

    def add_workflow(self, wf: T) -> T:
      self.all_workflows.append(wf)
      return wf

    def update_sub_workflows(self):
        for wf in self.all_workflows:
            wf.workflow_dir = self.workflow_dir/wf.workflow_dir.name

    def reset(self):
        for wf in self.all_workflows:
            wf.reset()

    def create_dirs(self, exclude_exprs=('.',)):
        super().create_dirs(exclude_exprs)
        for wf in self.all_workflows:
            wf.create_dirs(exclude_exprs)

    def run(self, annotation_path=FPIC_SMDS, full_images_dir=FPIC_IMAGES, label_map: pd.DataFrame=None):
        self.img_wf.run(annotation_path, full_images_dir)
        self.regionprop_wf.run(self.img_wf)
        self.img_export_wf.run(self.img_wf)
        self.tvt_wf.run(self.img_export_wf, label_map)
        self.link_wf.run(self.tvt_wf)

if __name__ == '__main__':
    mwf_folder = Path.home()/'Desktop/rgb_features_512'
    mwf = MainWorkflow(
        mwf_folder,
        comp_imgs_config=dict(shape=(512, 512)),
        create_dirs=True,
        # reset=True
    )
    mwf.run(
        # annotation_path=mwf.workflow_dir/'subset',
        label_map=pd.read_csv(mwf.workflow_dir / 'aliased_labels.csv', index_col=['numeric_label'])
    )
