from __future__ import annotations

import functools
import os
import pickle
import sys
from pathlib import Path

import cv2 as cv
import joblib
import numpy as np
import pandas as pd
import pyqtgraph as pg
from autobom.constants import TEMPLATES_DIR
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import FPIC_SMDS, FPIC_IMAGES
from s3a import TableData, ComponentIO, generalutils as gutils
from s3a.generalutils import pd_iterdict, resize_pad
from utilitys import fns
from utilitys.typeoverloads import FilePath
from utils import S3AFeatureWorkflow, WorkflowDir, RegisteredPath

DEFAULT_IMAGE_FEATURE_CONFIG = dict(
    shape=(50,50),
    keepAspectRatio=True,
)

# Override the Parallel class since there's no easy way to provide more informative print messages
class NamedParallel(joblib.Parallel):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        if name is None:
            name = str(self)
        self.name = name

    def _print(self, msg, msg_args):
        if not self.verbose:
            return
        if self.verbose < 50:
            writer = sys.stderr.write
        else:
            writer = sys.stdout.write
        msg = msg % msg_args
        writer('[%s]: %s\n' % (self.name, msg))

class CompImgsWorkflow(S3AFeatureWorkflow):
    """
    Prescribes a workflow for generating data sufficient for SMD designator descrimination unit.
    The final output is a pickled dataframe of component images as well as a populated directory with
    most intermediate steps, and some fitted data transformers. See `run` for more information
    """
    comp_imgs_dir = RegisteredPath()
    designator_labels_file = RegisteredPath('.csv')
    comp_imgs_file = RegisteredPath('.pkl')

    transformers_path = RegisteredPath()

    def __init__(self, workflow_folder: Path | str, config: dict=None, **kwargs):
        """
        Initializes the workflow with a location and set of data parameters

        :param workflow_folder: Where to generate the training outputs. Should either be empty or not exist to avoid accidentally
            overwriting any files
        :param config: Configuration for generating data. See ``DEFAULT_IMAGE_FEATURE_CONFIG`` for valid options
        """
        if config is None:
            config = DEFAULT_IMAGE_FEATURE_CONFIG
        super().__init__(workflow_folder, config=config, **kwargs)
        td = TableData(cfgDict=fns.attemptFileLoad(TEMPLATES_DIR/'proj_smd.s3aprj'))
        self.io = ComponentIO(td)
        self.desig_col = td.fieldFromName('Designator')
        # Can't do at class level since __set_name__ will be called
        self._converted_inputs_dir = self.comp_imgs_dir

    @functools.lru_cache()
    def create_get_designator_mapping(self):
        """
        Creates a complete list of all designators from the cleaned input
        """
        # Use an lru cache instead of checking for file
        # if self.designator_labels_file.exists():
        #     return pd.read_csv(
        #         self.designator_labels_file,
        #         dtype=str,
        #         na_filter=False,
        #         index_col='Numeric Label')['Designator']

        # Passing dataframes over multiprocessing is slower than re-reading a new file's dataframe each iteration.
        # So, just don't use this dataframe for anything other than getting unique designators
        designators = pd.concat([pd.read_csv(f, dtype=str, na_filter=False).Designator for f in self.formatted_input_path.glob('*.csv')])
        limits_counts = designators.value_counts()

        info_df = pd.DataFrame(np.c_[limits_counts.index, limits_counts.values], columns=['Designator', 'Count'])
        info_df.index.name = 'Numeric Label'
        info_df.index += 1
        info_df.to_csv(self.designator_labels_file)
        return info_df['Designator']

    def create_comp_imgs_df_single(self, file, src_dir, return_df=False):
        """
        Turns a csv annotation of a single image into a dataframe of cropped components from that image
        """
        name = file.name
        df = self.io.importCsv(file)
        df.columns[df.columns.get_loc(self.desig_col)].opts['limits'] = self.create_get_designator_mapping().to_list()
        exported = self.io.exportCompImgsDf(
            df,
            srcDir=src_dir,
            resizeOpts=self.config,
            labelField=self.desig_col,
        )
        self.maybe_reorient_comp_imgs(exported)
        # Unjumble row ordering
        col_order = ['instanceId', 'label', 'numericLabel', 'offset', 'rotated', 'image', 'labelMask' ]
        # Ensure nothing was lost in the reordering
        assert all(c in exported for c in col_order)
        assert all(c in col_order for c in exported)
        exported = exported[col_order]
        exported.to_pickle((self.comp_imgs_dir / name).with_suffix('.pkl'))
        if return_df:
            return exported

    def maybe_reorient_comp_imgs(self, df: pd.DataFrame, add_numeric_label=True):
        """
        Ensures the width of all SMD components is less than the height (i.e. every component will have a "vertical"
        alignment"
        """
        df['rotated'] = False
        if add_numeric_label:
            # Pre-set field to avoid errors setting single values later
            df['numericLabel'] = -1
        mapping = self.create_get_designator_mapping()
        for idx, row in pd_iterdict(df, index=True):
            numeric_lbl = mapping.index[np.argmax(mapping == row['label'])]
            bool_mask = row['labelMask'] == numeric_lbl
            on_pixs = np.nonzero(bool_mask)
            spans = [np.ptp(pixs) for pixs in on_pixs]
            # If width has a larger span than height, rotate so all components have the same preferred component
            # aspect
            if spans[1] > spans[0]:
                for kk in 'image', 'labelMask':
                    df.at[idx, kk] = cv.rotate(row[kk], cv.ROTATE_90_CLOCKWISE)
                df.at[idx, 'rotated'] = True
            if add_numeric_label:
                df.at[idx, 'numericLabel'] = numeric_lbl
        return df

    def create_all_comp_imgs(self, full_images_dir):
        # Save to intermediate directory first to avoid multiprocess comms bandwidth issues
        # delayed = joblib.delayed(self.create_single_image_features)
        # NamedParallel(verbose=10, n_jobs=8, name='Image Features')(delayed(file) for file in self.formatted_input_files)
        fns.mproc_apply(
            self.create_comp_imgs_df_single,
            self.new_input_files,
            src_dir=full_images_dir,
            descr='Creating image features',
            #debug=True
        )

    def merge_comp_imgs(self):
        """
        After single image features are created, this merges the results into a single dataframe for easier reading
        later
        """
        all_dfs = []
        for file in self.comp_imgs_dir.glob('*.pkl'):
            subdf = pd.read_pickle(file)
            subdf['imageFile'] = file.stem + '.png'
            all_dfs.append(subdf)
        feats_concat = pd.concat(all_dfs, ignore_index=True)
        feats_concat.to_pickle(self.comp_imgs_file)
        return feats_concat

    def get_feats_labels(self, df: pd.DataFrame, return_df=False):
        """
        Turns a dataframe with X rows of MxNx3 component images into a (X x M*N*3) 2 dimensional array where
        each pixel is a feature and each row is a new sample
        """
        feats = np.vstack(df['image'].apply(np.ndarray.ravel))
        desigs = self.create_get_designator_mapping()
        inverse = pd.Series(index=desigs.values, data=desigs.index)
        labels = inverse[df['label'].to_numpy()].values
        if return_df:
            return feats, labels, df
        return feats, labels


    def fit_save_transformer(self, transformer, feats_labels_or_df: pd.DataFrame | tuple):
        if not isinstance(feats_labels_or_df, tuple):
            feats_labels = self.get_feats_labels(feats_labels_or_df)
        else:
            feats_labels = feats_labels_or_df
        transformer.fit(*feats_labels)
        save_name = self.transformers_path / (type(transformer).__name__ + '.pkl')
        with open(save_name, 'wb') as ofile:
            pickle.dump(transformer, ofile)
        return transformer

    def fit_all_transformers(self):
        # Images are too big natively to fit into one dataframe, so resize them
        def read_func(file: Path):
            df_ = pd.read_pickle(file)
            for idx, img in df_['image'].iteritems():
                df_.at[idx, 'image'] = resize_pad(img, (50,50))
            return df_
        df = fns.readDataFrameFiles(self.comp_imgs_dir, read_func)
        # Ensure classes aren't overrepresented and blanks aren't an actual class
        df: pd.DataFrame = df[df['label'].str.len() > 0]
        df = df.groupby('label').apply(lambda el: el.sample(n=min(len(el), 150), random_state=10)).reset_index(drop=True)
        feats_labels = self.get_feats_labels(df)
        for xformer in tqdm([PCA(), LDA()], desc='Fitting transformers'):
            self.fit_save_transformer(xformer, feats_labels)


    def run(self, annotation_path=FPIC_SMDS, full_images_dir=FPIC_IMAGES):
        """
        Entry point for creating image features
        :param annotation_path: CSV image annotations to convert to comp imgs
        :param full_images_dir: Directory of full-sized PCB images. If *None*, defaults to FPIC pcb_image if FPIC
          is set on the system
        """
        self.create_formatted_inputs(annotation_path)

        self.create_all_comp_imgs(full_images_dir)
        # Can't merge with really big resize shape due to memory
        # merged = self.merge_comp_imgs()

        # self.fit_all_transformers()

class CompImgsExportWorkflow(WorkflowDir):

    TRAIN_NAME = 'train'
    VALIDATION_NAME = 'validation'
    TEST_NAME = 'test'

    ALL_DATA_TYPE_NAMES = [TRAIN_NAME, VALIDATION_NAME, TEST_NAME]

    images_dir = RegisteredPath()
    masks_dir = RegisteredPath()

    label_masks_dir = RegisteredPath(prefix=masks_dir, trim_exprs=('masks_dir',))
    rgb_masks_dir = RegisteredPath(prefix=masks_dir, trim_exprs=('masks_dir',))
    binary_masks_dir = RegisteredPath(prefix=masks_dir, trim_exprs=('masks_dir',))

    summaries_dir = RegisteredPath()
    summary_file = RegisteredPath('.html')

    def run(self, comp_imgs_wf: CompImgsWorkflow):
        """
        Automatically generates the Neural Network data in an appropriate directory structure
        and format in the base path with the resized and padded images and corresponding binary Masks.
        """
        files = comp_imgs_wf.comp_imgs_dir.glob('*.*')
        def checkdir(stem):
            for name in self.ALL_DATA_TYPE_NAMES:
                yield any((self.images_dir/name).glob(f'{stem}*'))
        new_files = fns.naturalSorted(f for f in files if not any(checkdir(f.stem)))

        mapping = comp_imgs_wf.create_get_designator_mapping()
        num_classes = len(mapping.index.unique())

        fns.mproc_apply(
            self._train_val_test_single,
            new_files,
            descr="Generating Dataset",
            showProgress=True,
            applyAsync=True,
            num_classes=num_classes,
            debug=True,
        )

        self.merge_summaries()

    def _train_val_test_single(self, comp_imgs_file, num_classes=None):
        out_df = pd.read_pickle(comp_imgs_file)
        out_df['imageFile'] = comp_imgs_file.stem
        # Dummy column which will be filled in during export
        out_df['dataType'] = ''

        datasets = {}
        test_pct = 0.15
        train_pct = 1-test_pct*2
        # Account for times where only 1-2 samples are in the imag annotation
        if int(train_pct) * len(out_df) < 1:
            # Randomly assign samples to train, validate, or test
            dest = np.random.choice(self.ALL_DATA_TYPE_NAMES)
            datasets[dest] = out_df
            for other in np.setdiff1d(self.ALL_DATA_TYPE_NAMES, [dest]):
                datasets[other] = pd.DataFrame()
        else:
            train_temp, datasets[self.TEST_NAME] = train_test_split(out_df, test_pct)
            datasets[self.TRAIN_NAME], datasets[self.VALIDATION_NAME] = train_test_split(train_temp, test_size=test_pct)

        for data_type, df in datasets.items():
            for dir_ in self.images_dir, self.label_masks_dir, self.rgb_masks_dir, self.binary_masks_dir:
                os.makedirs(dir_ / data_type, exist_ok=True)
            for index, row in gutils.pd_iterdict(df, index=True):
                update_keys = self._export_single_comp(index, row, data_type, num_classes)
                for kk, vv in update_keys.items():
                    out_df.at[index, kk] = vv

        out_df.to_csv(
            self.summaries_dir / comp_imgs_file.with_suffix('.csv').name
        )

    def _export_single_comp(self, index, row: dict, data_type: str, num_classes=None):
        export_name = f'{row["imageFile"]}_id_{index}.png'
        ret = {}
        self.generate_colored_mask(row['labelMask'], self.rgb_masks_dir/data_type/export_name, num_classes, 'viridis')
        self.generate_colored_mask(row['labelMask'], self.binary_masks_dir/data_type/export_name, num_classes, 'binary')


        for ret_key, dir_ in zip(
            ['image', 'labelMask'],
            [self.images_dir, self.label_masks_dir]
        ):
            save_name = dir_ / data_type / export_name
            gutils.cvImsave_rgb(save_name, row[ret_key])
            ret[ret_key] = gutils.imgPathtoHtml(save_name)

        ret["dataType"] = data_type

        return ret

    def merge_summaries(self):
        concat_df = fns.readDataFrameFiles(self.summaries_dir, pd.read_csv)
        gutils.pd_toHtmlWithStyle(concat_df, self.summary_file, style='img {width: 120px;}', escape=False, index=False)
        return concat_df

    @staticmethod
    def generate_colored_mask(
        label_mask: np.ndarray | FilePath,
        output_file,
        num_classes: int=None,
        color_map: str=None
    ):
        """
        A helper function that generates rescaled or RGB versions of label masks
        :param label_mask: numpy image to transform (or input file containing a mask)
        :param num_classes: The number of output labels for the Neural Network data. If not provided, defaults
          to the maximum value in the label mask
        :param output_file: Location to export the transformed image
        :param color_map: A string of the Matplotlib color map to use for the generated RGB ground truth segmentation masks.
            Acceptable color maps are restrained to the following:
             https://matplotlib.org/stable/tutorials/colors/colormaps.html. If *None*, uses the raw label mask without
             any changes. If "binary", turns any pixel > 0 to white and any pixel = 0 as black

        """
        if color_map is None:
            # Directly save the image
            gutils.cvImsave_rgb(output_file, label_mask)
            return
        if color_map == 'binary':
            # No need to use colormap -- just force high values on save
            # Values > 1 should all clip to 255
            gutils.cvImsave_rgb(output_file, (label_mask > 0).astype('uint8') * 255)
            return
        if isinstance(label_mask, FilePath.__args__):
            label_mask = gutils.cvImread_rgb(label_mask)
        if num_classes is None:
            num_classes = np.max(label_mask)
        all_colors = pg.colormap.get(color_map).getLookupTable(nPts=num_classes + 1)
        item = pg.ImageItem(label_mask, levels=[0, num_classes])
        item.setLookupTable(all_colors, update=True)
        item.save(str(output_file))
