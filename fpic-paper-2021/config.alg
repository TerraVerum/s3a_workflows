# Some locations have syntax errors so they must be filled in since they vary from host to host
primitive:
  datagen:
  - s3awfs.fmtinput.FormattedInputWorkflow:
      annotationsPath #: csv annotation file or folder
      augmentationOpts:
        fraction: 1.5
        rotationPct: 0.4
  - s3awfs.compimgs.ComponentImagesWorkflow:
      imagesPath #: Folder of full-size images
      labelField: Designator
      s3aProj #: S3A project that knows how to deserialize component data
      resizeOpts:
        shape: [512, 512]
  - s3awfs.png.PngExportWorkflow
  - s3awfs.tvtsplit.TrainValidateTestSplitWorkflow:
      labelMap: null # optional label alias file, must contain numeric_label and label columns
      valPct: 0.15
      testPct: 0.01 # Lots of unannotated images on which to test
      replace: false
      maxTestSamps: null
      maskColors: [ null ] # Don't generate overlays/rescaled masks to save time\

  LinkNet Training:
    - s3awfs.models.tensorflow.train.makeTensorflowStrategy
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
