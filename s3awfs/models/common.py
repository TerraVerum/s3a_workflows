from pathlib import Path

import cv2 as cv
import imageio
import numpy as np
import pyqtgraph as pg
from s3a import generalutils as gutils
from qtextras import widgets, RunOptions, fns, bindInteractorOptions as bind


class EpochEvolutionViewer(widgets.ImageViewer):
    def __init__(self, predictionsDir=None, **kwargs):
        super().__init__(**kwargs)
        self.predictionsDir = None
        self._epochDirs = []

        self.toolsEditor.registerFunction(self.saveEvolutionGif)
        self.changeDataProc = self.toolsEditor.registerFunction(
            self.changeImage, runOpts=RunOptions.ON_CHANGED
        )

        predProc = self.toolsEditor.registerFunction(
            self.changePredictionsDir, runOpts=RunOptions.ON_CHANGED
        )
        if predictionsDir is not None:
            predProc(predictionsDir=predictionsDir)

    @bind(selectedImages=dict(type="list"), selectedEpoch=dict(type="list"))
    def changeImage(
        self,
        selectedImage="",
        selectedEpoch="",
        # groundTruthMasks: FilePath=None,
    ):
        selectedEpoch = Path(selectedEpoch)
        image = selectedEpoch / selectedImage
        if not image.is_file():
            return
        self.setImage(image)

    def changePredictionsDir(self, predictionsDir: Path = ""):
        """
        :param predictionsDir:
        type: filepicker
        asFolder: True
        """
        self.predictionsDir = Path(predictionsDir)
        p = self.changeDataProc.parameters
        epochDirs = fns.naturalSorted(
            [d for d in self.predictionsDir.iterdir() if d.is_dir()]
        )
        p["selectedEpoch"].setLimits({d.name: d for d in epochDirs})
        uniqueImgs = {im.name for e in epochDirs for im in e.glob("*.*")}
        if any(epochDirs):
            p["selectedImage"].setLimits(fns.naturalSorted(uniqueImgs))
        else:
            p["selectedImage"].setLimits([])

    @bind(
        filename=dict(
            type="file",
            fileMode="AnyFile",
            value="predictions.gif",
            filter="Gif Files (*.gif);;",
        )
    )
    def saveEvolutionGif(self, filename=None, fps=2, epochAsText=True):
        p = self.changeDataProc.parameters
        ims = []
        selected = p["selectedImage"].value()
        drawitem = widgets.MaskCompositor()
        for epoch in p["selectedEpoch"].opts["limits"].values():
            imageFile = epoch / selected
            im = gutils.cvImreadRgb(imageFile)
            if im is None:
                continue
            if epochAsText:
                drawitem.clearOverlays()
                drawitem.setImage(im)
                drawitem.addMask(
                    np.zeros(im.shape[:2], "uint8"),
                    f"{imageFile.parent.name}/{imageFile.name}",
                )
                saved = drawitem.save()
                savedAsImg = saved.toImage()
                im = cv.cvtColor(
                    pg.imageToArray(savedAsImg, transpose=False), cv.COLOR_BGR2RGB
                )
            ims.append(im)
        if filename is not None:
            imageio.mimsave(filename, ims, fps=fps)
        return ims

    @classmethod
    def main(cls, predictionsDir=None):
        pg.mkQApp()
        eev = cls(predictionsDir=predictionsDir)
        eev.show_exec()
