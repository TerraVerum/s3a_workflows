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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from utilitys import fns

from src.utils import ExportDir, RegisteredPath

DEFAULT_IMAGE_FEATURE_CONFIG = dict(
    shape=(50,50),
    keepAspectRatio=True,

)

class ImageFeatureWorkflow(ExportDir):
    formatted_input_path = RegisteredPath()
    comp_imgs_path = RegisteredPath()
    designator_labels_file = RegisteredPath('.csv')
    comp_imgs_file = RegisteredPath('.pkl')
    comp_imgs_oriented_file = RegisteredPath('.pkl')

    def __init__(self, folder: Path | str, config: dict=None):
        super().__init__(folder)
        td = TableData(cfgDict=fns.attemptFileLoad(TEMPLATES_DIR/'proj_smd.s3aprj'))
        self.io = ComponentIO(td)
        self.desig_col = td.fieldFromName('Designator')

        if config is None:
            config = DEFAULT_IMAGE_FEATURE_CONFIG
        self.config = config

    @property
    def formatted_input_files(self):
        return list(self.formatted_input_path.glob('*.csv'))

    def create_formatted_inputs(self, df: pd.DataFrame):
        df['Designator'] = df['Designator'].str.strip().str.upper()
        for image, subdf in df.groupby(RTF.SRC_IMG_FILENAME.name): # type: str, pd.DataFrame
            SerialExporter.writeFile(self.formatted_input_path/(image + '.csv'), subdf, readonly=False)

    def create_get_designator_mapping(self):
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
        name = file.name
        df = self.io.importCsv(file)
        df.columns[df.columns.get_loc(self.desig_col)].opts['limits'] = self.create_get_designator_mapping().to_list()
        self.io.exportCompImgsDf(
            df,
            (self.comp_imgs_path / name).with_suffix('.pkl'),
            srcDir=Path.home()/'Dropbox (UFL)/Optical Images/FPIC/pcb_image',
            resizeOpts=self.config,
            labelField=self.desig_col,
        )

    def make_all_image_features(self):
        # Save to intermediate directory first to avoid multiprocess comms bandwidth issues
        fns.mproc_apply(self.make_single_image_features, self.formatted_input_files)

    def merge_comp_imgs(self):
        all_dfs = []
        for file in self.comp_imgs_path.glob('*.pkl'):
            subdf = pd.read_pickle(file)
            subdf['sourceFile'] = file.stem
            all_dfs.append(subdf)
        feats_concat = pd.concat(all_dfs, ignore_index=True)
        feats_concat.to_pickle(self.comp_imgs_file)
        return feats_concat

    def create_rotated_features(self, df=None):
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
        if df is None:
            df = pd.read_pickle(self.comp_imgs_oriented_file)
        feats = np.vstack(df['image'].apply(np.ndarray.ravel))
        desigs = self.create_get_designator_mapping()
        inverse = pd.Series(index=desigs.values, data=desigs.index)
        labels = inverse[df['label'].to_numpy()].values
        if return_df:
            return feats, labels, df
        return feats, labels

    def plot_model_rep(self, samples_df: pd.DataFrame, trained_model):
        """samples_df must resemble the return value of ``exportCompImgsDf`` with ('label', 'image') columns at least"""
        feats, _ = self.get_feats_labels(samples_df)
        compare = trained_model.inverse_transform(trained_model.predict(feats))
        cmp_images = []
        for cmp in compare:
            cmp_images.append(cmp.reshape(*self.config['shape'], -1))



    def make_image_lda_features(self, df=None):
        feats, labels, df = self.get_feats_labels(df, return_df=True)

        lda = LinearDiscriminantAnalysis()
        xformed = lda.fit_transform(feats, labels)
        return xformed

    def run(self, annotation_path, reset=False):
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
        oriented = self.create_rotated_features(merged)

        feats = self.make_image_lda_features(oriented)



if __name__ == '__main__':
    from utils import NComponentVisualizer
    from sklearn.decomposition import PCA
    workflow = ImageFeatureWorkflow(Path.home()/'Desktop/rgb_features')
    df = pd.read_pickle(workflow.comp_imgs_oriented_file)
    feats, _ = workflow.get_feats_labels(df)

    pca = PCA()
    pca.fit(feats)
    with open('pca.pkl', 'wb') as ofile:
        pickle.dump(pca, ofile)
    viz = NComponentVisualizer(df, pca)
