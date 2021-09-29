import typing as t
from pathlib import Path
import datetime

import pandas as pd
# Some systems need qt initialized before cv gets imported
import pyqtgraph as pg

pg.mkQApp()

from workflows.constants import FPIC_IMAGES, FPIC_SMDS, SMD_INIT_OPTS
from workflows.features import RegionPropertiesWorkflow, ComponentImagesWorkflow
from workflows.features.imagefeats import PngExportWorkflow, TrainValTestWorkflow
from workflows.utils import WorkflowDir

T = t.TypeVar('T')

class MainWorkflow(WorkflowDir):
    def __init__(
        self,
        folder,
        comp_imgs_config=None,
        train_linknet=True,
        **kwargs
    ):
        folder = Path(folder)
        # Init attribute so add_workflow can function
        self.workflow_dir = folder
        self.all_workflows = []
        # Just to make the function name shorter
        _ = self.add_workflow
        self.stage_counter = 1

        self.img_wf        = _(ComponentImagesWorkflow, config=comp_imgs_config, **SMD_INIT_OPTS, **kwargs)
        self.regionprop_wf = _(RegionPropertiesWorkflow, **kwargs)
        self.img_export_wf = _(PngExportWorkflow, **kwargs)
        self.tvt_wf        = _(TrainValTestWorkflow, **kwargs)
        if train_linknet:
            self.link_wf   = self.create_add_linknet(**kwargs)
        else:
            self.link_wf   = None

        super().__init__(folder, **kwargs)

    def create_add_linknet(self, **kwargs):
        # Defer to avoid tensorflow gpu initialization where possible
        from workflows.models.linknet import LinkNetTrainingWorkflow
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.add_workflow(LinkNetTrainingWorkflow, f' ({date_time})', **kwargs)

    def add_workflow(self, wf_class: t.Type[T], folder_suffix='', **wf_kwargs) -> T:
        base_path = self.workflow_dir
        folder = base_path/(f'{self.stage_counter}. {wf_class.name}' + folder_suffix)
        wf = wf_class(folder, **wf_kwargs)
        self.all_workflows.append(wf)
        self.stage_counter += 1
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

        if self.link_wf:
            self.link_wf.run(self.tvt_wf)

if __name__ == '__main__':
    mwf_folder = Path.home()/'Desktop/rgb_features_512'
    mwf = MainWorkflow(
        mwf_folder,
        create_dirs=True,
        # train_linknet=False,
        # reset=True
    )
    mwf.run(
        annotation_path=mwf.workflow_dir/'subset',
        label_map=pd.read_csv(mwf.workflow_dir / 'aliased_labels.csv', index_col=['numeric_label'])
    )
