from __future__ import annotations

import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from autobom.constants import TEMPLATES_DIR
from s3a import TableData, ComponentIO, REQD_TBL_FIELDS as RTF
from s3a.compio.exporters import SerialExporter
from s3a.compio.importers import SerialImporter
from s3a.generalutils import pd_iterdict
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm
from utilitys import fns

from src.constants import FPIC_FOLDER
from src.utils import ExportDir, RegisteredPath

DEFAULT_IMAGE_FEATURE_CONFIG = dict(
    shape=(50,50),
    keepAspectRatio=True,

)

class ImageFeatureWorkflow(ExportDir):
    """
    Prescribes a workflow for generating data sufficient for SMD designator descrimination unit.
    The final output is a pickled dataframe of component images as well as a populated directory with
    most intermediate steps, and some fitted data transformers. See `run` for more information
    """
    formatted_input_path = RegisteredPath()
    comp_imgs_path = RegisteredPath()
    designator_labels_file = RegisteredPath('.csv')
    comp_imgs_file = RegisteredPath('.pkl')
    comp_imgs_oriented_file = RegisteredPath('.pkl')

    transformers_path = RegisteredPath()

    def __init__(self, workflow_folder: Path | str, config: dict=None):
        """
        Initializes the workflow with a location and set of data parameters

        :param workflow_folder: Where to generate the training outputs. Should either be empty or not exist to avoid accidentally
            overwriting any files
        :param config: Configuration for generating data. See ``DEFAULT_IMAGE_FEATURE_CONFIG`` for valid options
        """
        super().__init__(workflow_folder)
        td = TableData(cfgDict=fns.attemptFileLoad(TEMPLATES_DIR/'proj_smd.s3aprj'))
        self.io = ComponentIO(td)
        self.desig_col = td.fieldFromName('Designator')

        if config is None:
            config = DEFAULT_IMAGE_FEATURE_CONFIG
        self.config = config

    @property
    def formatted_input_files(self):
        """
        Helper property to act like a list of files from the cleansed inputs
        """
        return list(self.formatted_input_path.glob('*.csv'))

    def create_formatted_inputs(self, df: pd.DataFrame):
        """
        Generates cleansed csv files from the raw input dataframe. Afterwards, saves annotations in files separated
        by image to allow multiprocessing on subsections of components
        """
        df['Designator'] = df['Designator'].str.strip().str.upper()
        for image, subdf in df.groupby(RTF.IMG_FILE.name): # type: str, pd.DataFrame
            SerialExporter.writeFile(self.formatted_input_path/(image + '.csv'), subdf, readonly=False)

    def create_get_designator_mapping(self):
        """
        Creates a complete list of all designators from the cleaned input
        """
        if self.designator_labels_file.exists():
            return pd.read_csv(
                self.designator_labels_file,
                dtype=str,
                na_filter=False,
                index_col='Numeric Label')['Designator']

        # Passing dataframes over multiprocessing is slower than re-reading a new file's dataframe each iteration.
        # So, just don't use this dataframe for anything other than getting unique designators
        designators = pd.concat([pd.read_csv(f, dtype=str, na_filter=False).Designator for f in self.formatted_input_files])
        limits_counts = designators.value_counts()

        info_df = pd.DataFrame(np.c_[limits_counts.index, limits_counts.values], columns=['Designator', 'Count'])
        info_df.index.name = 'Numeric Label'
        info_df.index += 1
        info_df.to_csv(self.designator_labels_file)
        return info_df['Designator']

    def make_single_image_features(self, file):
        """
        Turns a csv annotation of a single image into a dataframe of cropped components from that image
        """
        name = file.name
        df = self.io.importCsv(file)
        df.columns[df.columns.get_loc(self.desig_col)].opts['limits'] = self.create_get_designator_mapping().to_list()
        self.io.exportCompImgsDf(
            df,
            (self.comp_imgs_path / name).with_suffix('.pkl'),
            srcDir=FPIC_FOLDER/'pcb_image',
            resizeOpts=self.config,
            labelField=self.desig_col,
        )

    def make_all_image_features(self):
        # Save to intermediate directory first to avoid multiprocess comms bandwidth issues
        fns.mproc_apply(self.make_single_image_features, self.formatted_input_files)

    def merge_comp_imgs(self):
        """
        After single image features are created, this merges the results into a single dataframe for easier reading
        later
        """
        all_dfs = []
        for file in self.comp_imgs_path.glob('*.pkl'):
            subdf = pd.read_pickle(file)
            subdf['sourceFile'] = file.stem
            all_dfs.append(subdf)
        feats_concat = pd.concat(all_dfs, ignore_index=True)
        feats_concat.to_pickle(self.comp_imgs_file)
        return feats_concat

    def create_rotated_features(self, df=None):
        """
        Ensures the width of all SMD components is less than the height (i.e. every component will have a "vertical"
        alignment"
        """
        if df is None:
            df = pd.read_pickle(self.comp_imgs_file)
        else:
            df = df.copy()
        df['rotated'] = False
        mapping = self.create_get_designator_mapping()
        for idx, row in tqdm(pd_iterdict(df, index=True), desc='Reorienting', total=len(df)):
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
        df.to_pickle(self.comp_imgs_oriented_file)
        return df

    def get_feats_labels(self, df: pd.DataFrame=None, return_df=False):
        """
        Turns a dataframe with X rows of MxNx3 component images into a (X x M*N*3) 2 dimensional array where
        each pixel is a feature and each row is a new sample
        """
        if df is None:
            df = pd.read_pickle(self.comp_imgs_oriented_file)
        feats = np.vstack(df['image'].apply(np.ndarray.ravel))
        desigs = self.create_get_designator_mapping()
        inverse = pd.Series(index=desigs.values, data=desigs.index)
        labels = inverse[df['label'].to_numpy()].values
        if return_df:
            return feats, labels, df
        return feats, labels


    def fit_save_transformer(self, transformer, feats_labels=None):
        if feats_labels is None:
            feats_labels = self.get_feats_labels()
        transformer.fit(*feats_labels)
        save_name = self.transformers_path / (type(transformer).__name__ + '.pkl')
        with open(save_name, 'wb') as ofile:
            pickle.dump(transformer, ofile)
        return transformer

    def fit_all_transformers(self, feats_labels=None):
        if feats_labels is None:
            feats_labels = self.get_feats_labels()
        for xformer in PCA(), LDA():
            self.fit_save_transformer(xformer, feats_labels)


    def run(self, annotation_path, reset=False):
        """
        Top-level function. Takes either a csv file or folder of csvs and produces the final result. So, this method
        will show the order in which all processes should be run
        """
        self.make_dirs()
        if reset:
            self.reset()
        if annotation_path.is_dir():
            df = pd.concat(SerialImporter.readFile(p) for p in annotation_path.glob('*.csv'))
        else:
            df = SerialImporter.readFile(annotation_path)

        self.create_formatted_inputs(df)
        self.make_all_image_features()
        merged = self.merge_comp_imgs()
        self.create_rotated_features(merged)
        self.fit_all_transformers()


if __name__ == '__main__':
    workflow = ImageFeatureWorkflow(Path.home()/'Desktop/rgb_features')
    workflow.run(FPIC_FOLDER/'smd_annotation', reset=True)
