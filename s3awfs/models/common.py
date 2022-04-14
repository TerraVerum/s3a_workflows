from pathlib import Path

import cv2 as cv
import imageio
import numpy as np
import pyqtgraph as pg
from s3a import generalutils as gutils
from utilitys import widgets, RunOpts, fns


class EpochEvolutionViewer(widgets.ImageViewer):
    def __init__(self, predictionsDir=None, **kwargs):
        super().__init__(**kwargs)
        self.predictionsDir = None
        self._epochDirs = []

        self.toolsEditor.registerFunc(self.saveEvolutionGif)
        self.changeDataProc = self.toolsEditor.registerFunc(self.changeImage, runOpts=RunOpts.ON_CHANGED)

        predProc = self.toolsEditor.registerFunc(self.changePredictionsDir, runOpts=RunOpts.ON_CHANGED)
        if predictionsDir is not None:
            predProc(predictionsDir=predictionsDir)

    def changeImage(self,
                   selectedImage='',
                   selectedEpoch='',
                   # groundTruthMasks: FilePath=None,
                   ):
        """
        :param selectedImage:
        type: list
        :param selectedEpoch:
        type: list
        """
        selectedEpoch = Path(selectedEpoch)
        image = selectedEpoch/selectedImage
        if not image.is_file():
            return
        self.setImage(image)

    def changePredictionsDir(self, predictionsDir: Path=''):
        """
        :param predictionsDir:
        type: filepicker
        asFolder: True
        """
        self.predictionsDir = Path(predictionsDir)
        p = self.changeDataProc.input.params
        epochDirs = fns.naturalSorted([d for d in self.predictionsDir.iterdir() if d.is_dir()])
        p['selectedEpoch'].setLimits({d.name: d for d in epochDirs})
        uniqueImgs = {im.name for e in epochDirs for im in e.glob('*.*')}
        if any(epochDirs):
            p['selectedImage'].setLimits(fns.naturalSorted(uniqueImgs))
        else:
            p['selectedImage'].setLimits([])

    def saveEvolutionGif(self, filename=None, fps=2, epochAsText=True):
        """
        :param filename:
        type: filepicker
        existing: False
        value: 'predictions.gif'
        fileFilter: Gif Files (*.gif);;
        """
        p = self.changeDataProc.input.params
        ims = []
        selected = p['selectedImage'].value()
        drawitem = widgets.MaskCompositor()
        for epoch in p['selectedEpoch'].opts['limits'].values():
            imageFile = epoch/selected
            im = gutils.cvImreadRgb(imageFile)
            if im is None:
                continue
            if epochAsText:
                drawitem.clearOverlays()
                drawitem.setImage(im)
                drawitem.addMask(np.zeros(im.shape[:2], 'uint8'), f'{imageFile.parent.name}/{imageFile.name}')
                saved = drawitem.save()
                savedAsImg = saved.toImage()
                im = cv.cvtColor(pg.imageToArray(savedAsImg, transpose=False), cv.COLOR_BGR2RGB)
            ims.append(im)
        if filename is not None:
            imageio.mimsave(filename, ims, fps=fps)
        return ims

    @classmethod
    def main(cls, predictionsDir=None):
        pg.mkQApp()
        eev = cls(predictionsDir=predictionsDir)
        eev.show_exec()