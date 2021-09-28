from pathlib import Path

from constants import FPIC_IMAGES, FPIC_SMDS
from features.imagefeats import CompImgsExportWorkflow
from models.linknet import LinkNetWorkflow
from src.constants import FPIC_FOLDER
from src.features import RegionPropsWorkflow, CompImgsWorkflow
from utils import WorkflowDir, RegisteredPath

class MainWorkflow(WorkflowDir):
    linknet_dir = RegisteredPath()

    def run(self, annotation_path=FPIC_SMDS, full_images_dir=FPIC_IMAGES, comp_imgs_config=None, reset=False):
        opts = dict(create_dirs=True, reset=reset)
        img_wf = CompImgsWorkflow(self.workflow_dir, comp_imgs_config, **opts)
        regionprop_wf = RegionPropsWorkflow(self.workflow_dir, **opts)
        img_export_wf = CompImgsExportWorkflow(self.workflow_dir, **opts)
        #
        img_wf.run(annotation_path, full_images_dir)
        regionprop_wf.create_all_regionprops()
        img_export_wf.run(img_wf)

        link_wf = LinkNetWorkflow(self.linknet_dir)
        link_wf.run(img_export_wf, img_wf.create_get_designator_mapping())



if __name__ == '__main__':
    folder = Path.home()/'Desktop/rgb_features_512'
    wf = MainWorkflow(folder)
    wf.run(
        annotation_path=wf.workflow_dir/'formatted_input',
        comp_imgs_config=dict(shape=(512, 512))
    )
