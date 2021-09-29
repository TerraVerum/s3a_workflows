from pathlib import Path

import pandas as pd

from constants import FPIC_IMAGES, FPIC_SMDS
from features.imagefeats import CompImgsExportWorkflow, TrainValTestWorkflow
from models.linknet import LinkNetWorkflow
from src.features import RegionPropsWorkflow, CompImgsWorkflow
from utils import WorkflowDir

class MainWorkflow(WorkflowDir):
    def __init__(self, folder, comp_imgs_config=None, **kwargs):
        folder = Path(folder)
        self.img_wf = CompImgsWorkflow(
            folder/'1. Component Image Extraction',
            comp_imgs_config,
        )
        self.regionprop_wf = RegionPropsWorkflow(
            folder/'2. Region Props Extraction',
        )
        self.img_export_wf = CompImgsExportWorkflow(
            folder/'3. Raw Png Exports',
        )
        self.tvt_wf = TrainValTestWorkflow(
            folder / '4. Label Antialias + Train-Val-Test Split'
        )
        self.link_wf = LinkNetWorkflow(
            folder / '5. LinkNet Model Outputs'
        )

        self.all_workflows = [self.img_wf, self.regionprop_wf, self.img_export_wf, self.tvt_wf, self.link_wf]
        super().__init__(folder, **kwargs)

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
        # self.img_wf.run(annotation_path, full_images_dir)
        # self.regionprop_wf.run(self.img_wf)
        # self.img_export_wf.run(self.img_wf)
        # self.tvt_wf.run(self.img_export_wf, label_map)
        self.link_wf.run(self.tvt_wf)

if __name__ == '__main__':
    folder = Path.home()/'Desktop/rgb_features_512'
    mwf = MainWorkflow(
        folder,
        comp_imgs_config=dict(shape=(512, 512)),
        create_dirs=True
    )
    mwf.run(
        annotation_path=mwf.workflow_dir/'subset',
        label_map=pd.read_csv(mwf.workflow_dir / 'aliased_labels.csv', index_col=['numeric_label'])
    )
