from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, regionprops

from s3a import ComponentIO, REQD_TBL_FIELDS as RTF
from s3a.compio.helpers import deserialize
from qtextras import fns
from . import constants
from .fmtinput import FormattedInputWorkflow
from .utils import RegisteredPath, WorkflowDirectory

# -----
# Various image features
# -----
def aspect(img):
    return maxDim(img) / minDim(img)


def maxDim(img):
    return max(img.shape)


def minDim(img):
    return min(img.shape)


propNames = list(regionprops(np.ones((5, 5), int))[0]) + ["aspect"]
for unused in [
    "convex_image",
    "image_convex",
    "coords",
    "filled_image",
    "image_filled",
    "intensity_image",
    "image_intensity",
    "image",
    "centroid",
    "label",
    "moments",
    "slice",
]:
    if unused in propNames:
        # May or may not exist depending on scikit version
        propNames.remove(unused)


class RegionPropertiesWorkflow(WorkflowDirectory):
    io = ComponentIO()

    regionpropFeaturesFile = RegisteredPath(".csv")  # Concatenated features
    regionpropFeaturesDir = RegisteredPath()  # Per-image features

    def textAnnToRegionpropsCsv(self, annFile: Path, useFeatures=None, returnDf=False):
        """
        Creates regionprops features for an S3A annotation file. Must have a "vertices"
        column
        """
        if useFeatures is None:
            useFeatures = propNames
        df = self.io.importSerialized.readFile(
            annFile, usecols=["Vertices", "Instance ID"]
        )
        vertices, errs = deserialize(RTF.VERTICES, df["Vertices"])
        if len(errs):
            raise RuntimeError(
                f"Errors during import for {annFile.name}:\n" f"{errs.to_string()}"
            )
        if len(vertices) == 0:
            return
        offsets = np.vstack(vertices.apply(lambda el: el.stack().min(0)))
        # Compute on local coordinates by removing offsets for memory reasons
        # (out of memory error happened with enough files being multiprocessed)
        masks = vertices.apply(
            lambda el: el.removeOffset().toMask(asBool=False)
        ).to_list()

        if "aspect" in useFeatures:
            useFeatures.remove("aspect")
            extra = [aspect]
        else:
            extra = ()
        allProps = []
        for mask in masks:
            outDict = regionprops_table(
                mask, properties=useFeatures, extra_properties=extra
            )
            subdf = pd.DataFrame(outDict)
            allProps.append(subdf)

        propsDf = pd.concat(allProps)
        # monkey patch bbox since every coordinate is local
        propsDf[["bbox-0", "bbox-1"]] = offsets
        propsDf = propsDf.rename(
            columns={
                "bbox-0": "x",
                "bbox-1": "y",
                "bbox-2": "height",
                "bbox-3": "width",
            }
        )
        index = pd.MultiIndex.from_product(
            [[annFile.name], df["Instance ID"].to_numpy(object)],
            names=["Image File", "Instance ID"],
        )
        propsDf.index = index
        propsDf.to_csv(self.regionpropFeaturesDir / annFile.name)
        if returnDf:
            return propsDf

    @fns.dynamicDocstring(availableFeats=propNames)
    def runWorkflow(self, useFeatures: list[str] = None):
        """
        Creates a table of ``skimage.regionprops`` where each row corresponds to items
        from annotation vertices

        Parameters
        ----------
        useFeatures
            If given, only these features will be extracted from annotation masks. Can
            be a list with any of the following items (defaults to all if not provided)
            -- {availableFeats}
        """
        if useFeatures is None:
            useFeatures = propNames
        generated = {f.stem for f in self.regionpropFeaturesDir.glob("*.*")}
        newFiles = fns.naturalSorted(
            f
            for f in self.parent().get(FormattedInputWorkflow).formattedFiles
            if f.stem not in generated
        )
        fns.multiprocessApply(
            self.textAnnToRegionpropsCsv,
            newFiles,
            useFeatures=useFeatures,
            descr="Forming Region Properties",
            debug=constants.DEBUG,
        )
        df = pd.concat(
            [pd.read_csv(f) for f in self.regionpropFeaturesDir.glob("*.csv")]
        )
        df.to_csv(self.regionpropFeaturesFile, index=False)
        return df
