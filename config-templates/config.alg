primitive:
  datagen:
  - s3awfs.fmtinput.FormattedInputWorkflow:
      annotationsPath #: csv annotation file or folder
      augmentationOptions: null
#        value:
#          fraction: 0.0 # Will add "fraction*numrows" augmentations, may be >1
#          rotationPct: 0.4
  - s3awfs.compimgs.ComponentImagesWorkflow:
      imagesPath #: Folder of full-size images
      labelField: null #: Field to use as a color for mask export. Null is a binary mask
      s3aProject: null # S3A project that knows how to deserialize component data
      resizeOptions:
        value:
          shape: [512, 512]
  - s3awfs.png.PngExportWorkflow:
      overlayOptions: null # Set `value` to dict with "opacity" (between 0->1) and "colormap" keys
  - s3awfs.tvtsplit.TrainValidateTestSplitWorkflow:
      labelMap: null # optional label alias file, must contain numeric_label and label columns
      valPct: 0.15
      testPct: 0.01 # Lots of unannotated images on which to test
      replace: false
      maxTestSamps: null
      maskColors: [ null ] # Don't generate overlays/rescaled masks to save time

  Semantic LinkNet:
    - s3awfs.models.linknet.LinkNetTrainingWorkflow

top:
  Data Generation:
    - Create Directories
    - datagen
  Semantic LinkNet Training:
    - Create Directories
    - datagen:
      enabled: False
    - Semantic LinkNet
modules:
- s3awfs.utils
