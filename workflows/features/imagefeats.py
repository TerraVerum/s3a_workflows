from __future__ import annotations

import functools
import math
import os
import pickle
import shutil
import tempfile
import typing as t
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from s3a import TableData, ComponentIO, generalutils as gutils, REQD_TBL_FIELDS as RTF, ComplexXYVertices
from s3a.compio.exporters import SerialExporter
from s3a.compio.importers import SerialImporter
from s3a.generalutils import pd_iterdict, resize_pad
from utilitys import fns, PrjParam
from utilitys.typeoverloads import FilePath
from ..constants import FPIC_SMDS, FPIC_IMAGES
from ..utils import WorkflowDir, RegisteredPath, AliasedMaskResolver

class CompImgsWorkflow(WorkflowDir):
    """
    Prescribes a workflow for generating data sufficient for SMD label-based segmentation unit.
    The final output is a pickled dataframe of component images as well as a populated directory with
    most intermediate steps, and some fitted data transformers. See `run` for more information
    """
    formatted_input_path = RegisteredPath()

    comp_imgs_dir = RegisteredPath()
    all_labels_file = RegisteredPath('.csv')
    comp_imgs_file = RegisteredPath('.pkl')

    transformers_path = RegisteredPath()

    default_config = dict(
        shape=(512,512),
        keepAspectRatio=True,
    )

    @property
    def new_input_files(self):
        """
        Helper property to act like a list of files from the cleansed inputs
        """
        files = self.formatted_input_path.glob('*.csv')

        generated = {f.stem for f in self.comp_imgs_dir.glob('*.*')}
        return fns.naturalSorted(f for f in files if f.stem not in generated)

    def __init__(
        self,
        workflow_folder: Path | str,
        config: dict = None,
        s3a_proj: FilePath | dict = None,
        label_field: PrjParam = None,
        **kwargs
    ):
        """
        Initializes the workflow with a location and set of data parameters

        :param workflow_folder: Where to generate the training outputs. Should either be empty or not exist to avoid accidentally
            overwriting any files
        :param config: Configuration for generating data. See ``self.default_config`` for valid options
        :param prj_spec: S3A project file or dict for interpreting csvs
        :param label_field: Field to use as label moving forward
        """
        super().__init__(workflow_folder, config=config, **kwargs)
        if isinstance(s3a_proj, FilePath.__args__):
            s3a_proj = fns.attemptFileLoad(s3a_proj)

        td = TableData(cfgDict=s3a_proj)
        self.io = ComponentIO(td)
        self.label_field = td.fieldFromName(label_field)

    def create_formatted_inputs(self, annotation_path: FilePath=None):
        """
        Generates cleansed csv files from the raw input dataframe. Afterwards, saves annotations in files separated
        by image to allow multiprocessing on subsections of components
        """
        if annotation_path is None:
            return pd.DataFrame()
        if annotation_path.is_dir():
            df = fns.readDataFrameFiles(annotation_path, SerialImporter.readFile)
        else:
            df = SerialImporter.readFile(annotation_path)
        for image, subdf in df.groupby(RTF.IMG_FILE.name):
            newName = Path(image).with_suffix('.csv').name
            dest = self.formatted_input_path/newName
            if not dest.exists():
                SerialExporter.writeFile(dest, subdf, readonly=False)
        return df


    @functools.lru_cache()
    def create_get_label_mapping(self):
        """
        Creates a complete list of all labels from the cleaned input
        """
        # Use an lru cache instead of checking for file

        # Passing dataframes over multiprocessing is slower than re-reading a new file's dataframe each iteration.
        # So, just don't use this dataframe for anything other than getting unique labels
        labels_ser = pd.concat([pd.read_csv(f, dtype=str, na_filter=False)[str(self.label_field)]
                                 for f in self.formatted_input_path.glob('*.csv')]
                                )
        limits_counts = labels_ser.value_counts()

        info_df = pd.DataFrame(np.c_[limits_counts.index, limits_counts.values], columns=['label', 'count'])
        info_df.index.name = 'numeric_label'
        info_df.index += 1
        info_df.to_csv(self.all_labels_file)
        return info_df['label']

    def create_comp_imgs_df_single(self, file, src_dir, return_df=False):
        """
        Turns a csv annotation of a single image into a dataframe of cropped components from that image
        """
        name = file.name
        df = self.io.importCsv(file)
        df.columns[df.columns.get_loc(self.label_field)].opts['limits'] = self.create_get_label_mapping().to_list()
        exported = self.io.exportCompImgsDf(
            df,
            srcDir=src_dir,
            resizeOpts=self.config,
            labelField=self.label_field,
        )
        self.maybe_reorient_comp_imgs(exported, df[RTF.VERTICES])
        # Unjumble row ordering
        col_order = ['instanceId', 'label', 'numericLabel', 'offset', 'rotated', 'image', 'labelMask' ]
        # Ensure nothing was lost in the reordering
        assert all(c in exported for c in col_order)
        assert all(c in col_order for c in exported)
        exported = exported[col_order]
        exported.to_pickle((self.comp_imgs_dir / name).with_suffix('.pkl'))
        if return_df:
            return exported

    def maybe_reorient_comp_imgs(
        self,
        df: pd.DataFrame,
        vertices: t.Sequence[ComplexXYVertices],
        add_numeric_label=True
    ):
        """
        Ensures the width of all SMD components is less than the height (i.e. every component will have a "vertical"
        alignment"
        """
        df['rotated'] = False

        mapping = self.create_get_label_mapping()
        numeric_labels = []
        for (idx, row), verts in zip(pd_iterdict(df, index=True), vertices):
            numeric_lbl = mapping.index[np.argmax(mapping == row['label'])]
            xy_span = np.ptp(verts.stack(), axis=0)
            numeric_labels.append(numeric_lbl)
            # If width has a larger span than height, rotate so all components have the same preferred component
            # aspect
            if xy_span[0] > xy_span[1]:
                for kk in 'image', 'labelMask':
                    df.at[idx, kk] = cv.rotate(row[kk], cv.ROTATE_90_CLOCKWISE)
                df.at[idx, 'rotated'] = True
        if add_numeric_label:
            df['numericLabel'] = numeric_labels
        return df

    def create_all_comp_imgs(self, full_images_dir):
        # Save to intermediate directory first to avoid multiprocess comms bandwidth issues
        # delayed = joblib.delayed(self.create_comp_imgs_df_single)
        # NamedParallel(verbose=10, n_jobs=8, name='Image Features')(
        #     delayed(file, src_dir=full_images_dir) for file in self.new_input_files
        # )
        fns.mproc_apply(
            self.create_comp_imgs_df_single,
            self.new_input_files,
            src_dir=full_images_dir,
            descr='Creating component image pickle files',
            # debug=True
        )

    def merge_comp_imgs(self):
        """
        After single image features are created, this merges the results into a single dataframe for easier reading
        later
        """
        all_dfs = []
        for file in self.comp_imgs_dir.glob('*.pkl'):
            subdf = pd.read_pickle(file)
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
        labels = self.create_get_label_mapping()
        inverse = pd.Series(index=labels.values, data=labels.index)
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


class PngExportWorkflow(WorkflowDir):

    TRAIN_NAME = 'train'
    VALIDATION_NAME = 'validation'
    TEST_NAME = 'test'

    ALL_DATA_TYPE_NAMES = [TRAIN_NAME, VALIDATION_NAME, TEST_NAME]

    images_dir = RegisteredPath()

    label_masks_dir = RegisteredPath()

    summaries_dir = RegisteredPath()
    summary_file = RegisteredPath('.csv')

    def run(self, comp_imgs_wf: CompImgsWorkflow):
        """
        Automatically generates the Neural Network data in an appropriate directory structure
        and format in the base path with the resized and padded images and corresponding binary Masks.
        """
        files = np.array(list(comp_imgs_wf.comp_imgs_dir.glob('*.*')))
        stems = [f.stem for f in files]
        if self.summary_file.exists():
            summary = pd.read_csv(self.summary_file)
            new_files = fns.naturalSorted(files[np.isin(stems, [Path(f).stem for f in summary['imageFile']], invert=True)])
        else:
            new_files = fns.naturalSorted(files)


        fns.mproc_apply(
            self._export_single_pcb_image,
            new_files,
            descr="Exporting Png Files",
            showProgress=True,
            applyAsync=True,
            # debug=True,
        )

        self.create_merged_summaries()

    def _export_single_pcb_image(self, comp_imgs_file):
        out_df = pd.read_pickle(comp_imgs_file)
        out_df['imageFile'] = comp_imgs_file.with_suffix('.png').name

        exported_imgs = []
        for index, row in gutils.pd_iterdict(out_df, index=True):
            image_name = self._export_single_comp(index, row)
            exported_imgs.append(image_name)
        out_df['compImageFile'] = exported_imgs
        out_df: pd.DataFrame
        out_df.drop(columns=['labelMask', 'image']).to_csv(
            self.summaries_dir / comp_imgs_file.with_suffix('.csv').name,
            index=False
        )

    def _export_single_comp(self, index, row: dict):
        export_name = f'{os.path.splitext(row["imageFile"])[0]}_id_{index}.png'

        for ret_key, dir_ in zip(
            ['image', 'labelMask'],
            [self.images_dir, self.label_masks_dir]
        ):
            save_name = dir_/export_name
            gutils.cvImsave_rgb(save_name, row[ret_key])

        return export_name

    def create_merged_summaries(self):
        concat_df = fns.readDataFrameFiles(self.summaries_dir, pd.read_csv)
        concat_df.to_csv(self.summary_file, index=False)
        return concat_df


class LabelMaskResolverWorkflow(WorkflowDir):
    """
    Turns masks with potentially many-to-one label mappings to sequentially numbered output values
    """
    rgb_masks_dir = RegisteredPath()
    binary_masks_dir = RegisteredPath()
    label_masks_dir = RegisteredPath()

    def run(
        self,
        label_mask_files: t.List[Path | np.ndarray],
        resolver: AliasedMaskResolver,
    ):
        for filename in label_mask_files:
            mask = resolver.get_maybe_resolve(filename)
            # Fetch out here to avoid fetching inside loop
            for cmap, dir_ in zip(
                [None, 'binary', 'viridis'],
                [self.label_masks_dir, self.binary_masks_dir, self.rgb_masks_dir]
            ):
                resolver.generate_colored_mask(mask, dir_/filename.name, resolver.num_classes, cmap, resolve=False)


class TrainValTestWorkflow(WorkflowDir):
    resolver = AliasedMaskResolver()

    TRAIN_NAME = 'train'
    VALIDATION_NAME = 'val'
    TEST_NAME = 'test'


    train_dir = RegisteredPath()
    val_dir = RegisteredPath()
    test_dir = RegisteredPath()

    filtered_summary_file = RegisteredPath('.csv')
    class_info_file = RegisteredPath('.csv')

    default_config = dict(
        balance_classes=True,
        balance_func='median',
        replace_samps=False,
        test_pct=0.15,
    )

    def run(self, export_wf: PngExportWorkflow, label_info_df: pd.DataFrame=None):

        summary = self.create_get_filtered_summary_df(
            pd.read_csv(export_wf.summary_file),
            label_info_df
        )
        self.resolver = AliasedMaskResolver(label_info_df['label'])
        self.resolver.class_info.to_csv(self.class_info_file)

        train_set = {self.train_dir: summary[summary['dataType'] == 'train']}
        other_set = {}
        for dir_, typ in zip([self.val_dir, self.test_dir], [self.VALIDATION_NAME, self.TEST_NAME]):
            other_set[dir_] = summary[summary['dataType'] == typ]

        fns.mproc_apply(
            self._export_datatype_portion,
            (train_set, other_set),
            extraArgs=(export_wf,),
            descr='Forming Train/Val/Test Sets',
            # debug=True
        )

    def _export_datatype_portion(self, dir_summary_map: dict, export_wf):
        EW = PngExportWorkflow
        link_func = self._get_link_func()
        for dest_dir, df in dir_summary_map.items():
            mask_wf = LabelMaskResolverWorkflow(dest_dir, create_dirs=True)
            # Mask_wf generates colored, normal, and binary scaled masks
            mask_wf.run(
                df['compImageFile'].apply(
                    lambda el: export_wf.label_masks_dir / el),
                self.resolver
            )
            image_dir = dest_dir / EW.images_dir
            image_dir.mkdir(exist_ok=True)
            for row in pd_iterdict(df):
                comp_file = row['compImageFile']
                link_func(export_wf.images_dir / comp_file, image_dir / comp_file)

    def create_get_filtered_summary_df(self, summary_df, label_info_df):
        summary_df = self._filter_by_label(summary_df, label_info_df)
        if self.config.get('balance_classes'):
            summary_df = self._balance_classes(summary_df)
        summary_df = self._add_train_val_test_info(summary_df)

        summary_df.to_csv(self.filtered_summary_file, index=False)
        return summary_df

    @staticmethod
    def _filter_by_label(summary_df, label_info_df):
        if label_info_df is None:
            # Valid if any label is present
            membership = summary_df['label'].notnull().to_numpy(bool)
        else:
            membership = np.isin(summary_df['numericLabel'], label_info_df.index)
        # Only train on expected labels
        summary_df = summary_df[membership]
        return summary_df

    def _balance_classes(self, summary_df):
        grouped = summary_df.groupby('label')
        samp_size = math.ceil(grouped.size().apply(self.config['balance_func']))
        if self.config.get('replace'):
            sampler = lambda el: el.sample(n=samp_size, replace=True)
        else:
            sampler = lambda el: el.sample(n=min(len(el), samp_size))
        summary_df = grouped.apply(sampler).droplevel('label')
        return summary_df

    def _add_train_val_test_info(self, summary_df):
        test_pct = self.config['test_pct']
        train_temp, test_ids = train_test_split(summary_df.index, test_size=test_pct)
        _, val_ids = train_test_split(train_temp, test_size=test_pct)
        summary_df['dataType'] = self.TRAIN_NAME
        summary_df.loc[test_ids, 'dataType'] = self.TEST_NAME
        summary_df.loc[val_ids, 'dataType'] = self.VALIDATION_NAME
        return summary_df

    @staticmethod
    def _get_link_func():
        """
        Symlinks rarely have permission by default on windows so be able to copy if needed
        """
        # Use symlinks to avoid lots of file duplication
        try:
            link_func = os.symlink
            with tempfile.TemporaryDirectory() as td:
                src: Path = Path(td) / 'test'
                src.touch()
                link_func(src, src.with_name('testlink'))
        except (PermissionError, OSError):
            link_func = shutil.copy
        return link_func


    def create_dirs(self,exclude_exprs=('.',)):
        super().create_dirs(exclude_exprs)
        for sub in PngExportWorkflow.images_dir, PngExportWorkflow.label_masks_dir:
            for parent in self.train_dir, self.val_dir, self.test_dir:
                to_create = parent/sub
                if any(ex in str(to_create) for ex in exclude_exprs):
                    continue
                to_create.mkdir(exist_ok=True)
