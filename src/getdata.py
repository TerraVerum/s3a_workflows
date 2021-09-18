from pathlib import Path

import numpy as np
import pandas as pd
from s3a import ComplexXYVertices, ComponentIO
from s3a.parameditors.table import TableData
from autobom.constants import TEMPLATES_DIR

# FPIC_FOLDER = Path('/shared/FPIC/')
from skimage.measure import regionprops_table, regionprops
from tqdm import tqdm
from utilitys import fns

FPIC_FOLDER = Path.home()/'Dropbox (UFL)/Optical Images/FPIC'
SMD_FOLDER = FPIC_FOLDER/'smd_annotation'

def read_cleaned_annotation(ann_file, designator_count=2, return_blanks=False):
  """
  From all SMD data in FPIC, return the subset that
  
    - Is not a duplicate image (i.e. only cc 4 and 11-inf
    - Optionally has a designator
    - Can threshold to only return designators with a certain count
  """
  df = pd.read_csv(ann_file, dtype=str, na_filter=False)
  ccnum = df['Source Image Filename'].apply(lambda el: int(el.split('_')[3])).values
  # Keep only the ideal subset of data, i.e. ignore the environmental illumination study items
  non_duplicate = (ccnum == 4) | (ccnum > 10)
  subset = df[non_duplicate]

  # Get unique designators to threshold by count, etc.
  desigs, counts = np.unique(subset['Designator'], return_counts=True)
  valid_desigs = desigs[counts >= designator_count]
  subset = subset[np.isin(subset['Designator'], valid_desigs)]
  if not return_blanks:
    subset = subset[subset['Designator'].str.len() > 0]
  io = ComponentIO(TableData(TEMPLATES_DIR/'proj_smd.s3aprj'))
  return io.importCsv(subset).rename(columns=str)

def aspect(img):
  return max_dim(img)/min_dim(img)
def max_dim(img):
  return max(img.shape)
def min_dim(img):
  return min(img.shape)

def create_feature_df(annotations: pd.DataFrame, save_filename=None):
  """
  Consolidates all computed features into one dataframe, including all original annotation columns
  """
  if len(annotations) == 0:
    return
  masks = annotations['Vertices'].apply(lambda el: el.toMask(asBool=False)).to_list()

  # Can't easily choose "all properties" in table version, so get a sample of available
  # properties here
  prop_names = list(regionprops(masks[0])[0])
  for unused in ['convex_image', 'coords', 'filled_image', 'image']:
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
  if save_filename is not None:
    props_df.to_csv(save_filename, index=False)
  return props_df

def single_iter(file, out_folder, **kwargs):
  df = read_cleaned_annotation(file, **kwargs)
  create_feature_df(df, out_folder/file.name)

def create_all_annotation_features(out_folder):
  files = list(SMD_FOLDER.glob('*.csv'))
  fns.mproc_apply(single_iter, files, out_folder=out_folder)

def create_all_cleaned_annotations(out_file):
  files = list(SMD_FOLDER.glob('*.csv'))
  df = pd.concat(fns.mproc_apply(single_iter, files, return_blanks=True, designator_counts=0))

if __name__ == '__main__':
    out_dir = Path.home()/'Desktop/smd_features/'
    out_dir.mkdir(exist_ok=True)
    create_all_annotation_features(out_dir)
    brep = 1
