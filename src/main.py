from pathlib import Path

from src.constants import FPIC_FOLDER
from src.regionpropfeats import RegionPropsWorkflow
from src.imagefeats import ImageFeatureWorkflow

if __name__ == '__main__':
    folder = Path.home()/'Desktop/rgb_features'
    w1 = ImageFeatureWorkflow(folder)
    w2 = RegionPropsWorkflow(folder)
    # w1.run(FPIC_FOLDER/'smd_annotation', reset=True)
    w2.create_dirs()
    w2.feature_workflow()
