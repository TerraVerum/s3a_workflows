from __future__ import annotations
import typing as t
from s3a import TableData, ComponentIO, REQD_TBL_FIELDS as RTF
import pandas as pd
import numpy as np
from s3a.compio.exporters import SerialExporter
from s3a.compio.importers import SerialImporter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utilitys import fns
from pathlib import Path
from autobom.constants import TEMPLATES_DIR

from src.utils import ExportDir, RegisteredPath

class ImageFeatureWorkflow(ExportDir):
    formatted_input_path = RegisteredPath()
    comp_imgs_path = RegisteredPath()
    designator_labels_file = RegisteredPath('.csv')
    comp_imgs_file = RegisteredPath('.pkl')

    def __init__(self, folder: Path | str):
        super().__init__(folder)
        td = TableData(cfgDict=fns.attemptFileLoad(TEMPLATES_DIR/'proj_smd.s3aprj'))
        self.io = ComponentIO(td)
        self.desig_col = td.fieldFromName('Designator')

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
            resizeOpts=dict(shape=(50,50)),
            labelField=self.desig_col
        )

    def make_all_image_features(self):
        # Save to intermediate directory first to avoid multiprocess comms bandwidth issues
        fns.mproc_apply(self.make_single_image_features, self.formatted_input_files)

    def merge_comp_imgs(self):
        feats_concat = pd.concat([pd.read_pickle(f) for f in self.comp_imgs_path.glob('*.pkl')])
        feats_concat.to_pickle(self.comp_imgs_file)
        return feats_concat

    def make_image_ldafeatures(self, df=None):
        if df is None:
            df = pd.read_pickle(self.comp_imgs_file)
        feats = np.vstack(df.image.apply(np.ndarray.ravel))
        desigs = self.create_get_designator_mapping()
        inverse = pd.Series(index=desigs.values, data=desigs.index)
        labels = inverse[df.label.to_numpy()]

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
        feats = self.make_image_pca_features(merged)


if __name__ == '__main__':
    workflow = ImageFeatureWorkflow(Path.home()/'Desktop/rgb_features')
    workflow.make_image_pca_features()
