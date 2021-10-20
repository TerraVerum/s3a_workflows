from __future__ import annotations

from pathlib import Path

import pandas as pd
from s3a import REQD_TBL_FIELDS as RTF
from s3a.compio import SerialExporter, SerialImporter
from utilitys import fns
from utilitys.typeoverloads import FilePath

from .utils import WorkflowDir, RegisteredPath


class FormattedInputWorkflow(WorkflowDir):
  formattedInputPath = RegisteredPath()

  def runWorkflow(self, annotationPath: FilePath = None):
    """
    Generates cleansed csv files from the raw input dataframe. Afterwards, saves annotations in files separated
    by image to allow multiprocessing on subsections of components
    :param annotationPath: Can either be a file or folder path. These are the annotations that will be processed
      during the workflow.
    """
    if annotationPath is None:
      return pd.DataFrame()
    annotationPath = Path(annotationPath)
    if annotationPath.is_dir():
      df = fns.readDataFrameFiles(annotationPath, SerialImporter.readFile)
    else:
      df = SerialImporter.readFile(annotationPath)
    for image, subdf in df.groupby(RTF.IMG_FILE.name):
      newName = Path(image).with_suffix('.csv').name
      dest = self.formattedInputPath / newName
      if not dest.exists():
        SerialExporter.writeFile(dest, subdf, readonly=False)
    return df

  @property
  def formattedFiles(self):
      return fns.naturalSorted(self.formattedInputPath.glob('*.csv'))
