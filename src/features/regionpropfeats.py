from pathlib import Path

import numpy as np
import pandas as pd
from autobom.constants import TEMPLATES_DIR
from s3a import ComponentIO, REQD_TBL_FIELDS as RTF
from s3a.compio.helpers import deserialize
from s3a.parameditors.table import TableData
from skimage.measure import regionprops_table, regionprops
from utilitys import fns

from src.constants import FPIC_FOLDER
from src.utils import RegisteredPath, S3AFeatureWorkflow

SMD_FOLDER = FPIC_FOLDER/'smd_annotation'

# -----
# Various image features
# -----
def aspect(img):
    return max_dim(img)/min_dim(img)
def max_dim(img):
    return max(img.shape)
def min_dim(img):
    return min(img.shape)


class RegionPropsWorkflow(S3AFeatureWorkflow):
    io = ComponentIO(TableData(TEMPLATES_DIR/'proj_smd.s3aprj'))

    regionprop_features_file = RegisteredPath('.csv') # Concatenated features
    regionprop_features_dir = RegisteredPath() # Per-image features

    def text_ann_to_regionprops_csv(self, ann_file: Path, return_df=True):
        """
        Creates regionprops features for an S3A annotation file. Must have a "vertices" column
        """
        df = self.io.importSerialized.readFile(ann_file, usecols=['Vertices', 'Instance ID'])
        vertices, errs = deserialize(RTF.VERTICES, df['Vertices'])
        if len(errs):
            raise RuntimeError(f'Errors during import for {ann_file.name}:\n'
                               f'{errs.to_string()}')
        if len(vertices) == 0:
            return
        offsets = np.vstack(vertices.apply(lambda el: el.stack().min(0)))
        # Compute on local coordinates by removing offsets for memory reasons
        # (out of memory error happened with enough files being multiprocessed)
        masks = vertices.apply(lambda el: el.removeOffset().toMask(asBool=False)).to_list()

        # Can't easily choose "all properties" in table version, so get a sample of available
        # properties here
        prop_names = list(regionprops(masks[0])[0])
        for unused in ['convex_image', 'coords', 'filled_image', 'image', 'centroid', 'label', 'moments', 'slice']:
            prop_names.remove(unused)
        all_props = []
        for mask in masks:
            out_dict = regionprops_table(
                mask,
                properties=prop_names,
                extra_properties=(aspect, max_dim, min_dim)
            )
            subdf = pd.DataFrame(out_dict)
            all_props.append(subdf)

        props_df = pd.concat(all_props)
        # monkey patch bbox since every coordinate is local
        props_df[['bbox-0', 'bbox-1']] = offsets
        props_df = props_df.rename(columns={'bbox-0': 'x', 'bbox-1': 'y', 'bbox-2': 'height', 'bbox-3': 'width'})
        index = pd.MultiIndex.from_product([[ann_file.name], df['Instance ID'].to_numpy(object)],
                                           names=['Image File', 'Instance ID'])
        props_df.index = index
        props_df.to_csv(self.regionprop_features_dir / ann_file.name)
        if return_df:
            return props_df

    def create_all_regionprops(self):
        fns.mproc_apply(self.text_ann_to_regionprops_csv, self.new_input_files, return_df=False)
        # Concat after to avoid multiproc bandwidth
        df = fns.readDataFrameFiles(self.regionprop_features_dir, pd.read_csv)
        df.to_csv(self.regionprop_features_file, index=False)
        return df

    def run(self, annotation_path):
        """
        Top-level function. Takes either a csv file or folder of csvs and produces the final result. So, this method
        will show the order in which all processes should be run
        """
        self.create_formatted_inputs(annotation_path)
        self.create_all_regionprops()
