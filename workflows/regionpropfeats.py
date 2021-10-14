from pathlib import Path

import numpy as np
import pandas as pd
from autobom.constants import TEMPLATES_DIR
from s3a import ComponentIO, REQD_TBL_FIELDS as RTF
from s3a.compio.helpers import deserialize
from s3a.parameditors.table import TableData
from skimage.measure import regionprops_table, regionprops
from utilitys import fns

from . import constants
from .compimgs import ComponentImagesWorkflow
from .utils import RegisteredPath, NestedWorkflow, WorkflowDir


# -----
# Various image features
# -----
def aspect(img):
    return maxDim(img)/minDim(img)
def maxDim(img):
    return max(img.shape)
def minDim(img):
    return min(img.shape)

class RegionPropertiesWorkflow(WorkflowDir):
    io = ComponentIO(TableData(TEMPLATES_DIR/'proj_smd.s3aprj'))

    regionpropFeaturesFile = RegisteredPath('.csv') # Concatenated features
    regionpropFeaturesDir = RegisteredPath() # Per-image features

    def textAnnToRegionpropsCsv(self, annFile: Path, returnDf=True):
        """
        Creates regionprops features for an S3A annotation file. Must have a "vertices" column
        """
        df = self.io.importSerialized.readFile(annFile, usecols=['Vertices', 'Instance ID'])
        vertices, errs = deserialize(RTF.VERTICES, df['Vertices'])
        if len(errs):
            raise RuntimeError(f'Errors during import for {annFile.name}:\n'
                               f'{errs.to_string()}')
        if len(vertices) == 0:
            return
        offsets = np.vstack(vertices.apply(lambda el: el.stack().min(0)))
        # Compute on local coordinates by removing offsets for memory reasons
        # (out of memory error happened with enough files being multiprocessed)
        masks = vertices.apply(lambda el: el.removeOffset().toMask(asBool=False)).to_list()

        # Can't easily choose "all properties" in table version, so get a sample of available
        # properties here
        propNames = list(regionprops(masks[0])[0])
        for unused in ['convex_image', 'coords', 'filled_image', 'image', 'centroid', 'label', 'moments', 'slice']:
            propNames.remove(unused)
        allProps = []
        for mask in masks:
            outDict = regionprops_table(
                mask,
                properties=propNames,
                extra_properties=(aspect, maxDim, minDim)
            )
            subdf = pd.DataFrame(outDict)
            allProps.append(subdf)

        propsDf = pd.concat(allProps)
        # monkey patch bbox since every coordinate is local
        propsDf[['bbox-0', 'bbox-1']] = offsets
        propsDf = propsDf.rename(columns={'bbox-0': 'x', 'bbox-1': 'y', 'bbox-2': 'height', 'bbox-3': 'width'})
        index = pd.MultiIndex.from_product([[annFile.name], df['Instance ID'].to_numpy(object)],
                                           names=['Image File', 'Instance ID'])
        propsDf.index = index
        propsDf.to_csv(self.regionpropFeaturesDir / annFile.name)
        if returnDf:
            return propsDf

    def runWorkflow(self, parent: NestedWorkflow):
        """
        Top-level function. Takes either a csv file or folder of csvs and produces the final result. So, this method
        will show the order in which all processes should be run
        """
        compImgsWf = parent.get(ComponentImagesWorkflow)
        fns.mproc_apply(
            self.textAnnToRegionpropsCsv,
            fns.naturalSorted(compImgsWf.formattedInputPath.glob('*.csv')),
            returnDf=False,
            descr='Forming Region Properties',
            debug=constants.DEBUG
        )
        # Concat after to avoid multiproc bandwidth
        df = fns.readDataFrameFiles(self.regionpropFeaturesDir, pd.read_csv)
        df.to_csv(self.regionpropFeaturesFile, index=False)
        return df
