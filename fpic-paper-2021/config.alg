primitive:
  datagen:
  - s3awfs.fmtinput.FormattedInputWorkflow:
      annotationsPath: C:/Users/njessurun/Dropbox (UFL)/Optical Images/FPIC/smd_annotation/
      augmentationOpts:
        fraction: 1.5
        rotationPct: 0.4
  - s3awfs.compimgs.ComponentImagesWorkflow:
      imagesPath: C:/Users/njessurun/Dropbox (UFL)/Optical Images/FPIC/pcb_image
      labelField: Designator
      s3aProj: C:/Users/njessurun/Desktop/git/autobom/autobom/projtemplates/proj_smd.s3aprj
      resizeOpts:
        shape: [512, 512]
  - s3awfs.png.PngExportWorkflow
  - s3awfs.tvtsplit.TrainValidateTestSplitWorkflow:
      labelMap: null #/home/ntjess/Desktop/datagen/datagen/all_labels.csv
      valPct: 0.15
      testPct: 0.01 # Lots of unannotated images on which to test
      replace: false
      maxTestSamps: null
      maskColors: [ null ] # Don't generate overlays/rescaled masks to save time\

  LinkNet Training:
    - s3awfs.models.linknet.makeLinknetModel
    - s3awfs.models.tensorflow.train.TensorflowTrainingWorkflow:
      learningRate: 0.001
      batchSize: 12
      epochs: 1000
      tensorboardUpdatesPerEpoch: 0
      predictionDuringTrainPath:
      computeDevices:
        - /gpu:1
top:
  Data Generation:
    - Create Directories
    - datagen
  Semantic LinkNet Training:
    - Create Directories
    - datagen
    - LinkNet Training
  Bbox LinkNet Training:
    - Create Directories
    - datagen
    - LinkNet Training:
        convertMasksToBbox: True
        computeDevices:
          - /gpu:0
modules:
- s3awfs.utils
