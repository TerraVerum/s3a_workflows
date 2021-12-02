primitive:
  datagen:
  - s3awfs.fmtinput.FormattedInputWorkflow:
      annotationsPath: C:\Users\njessurun\Dropbox (UFL)\Optical Images\FPIC\smd_annotation\pcb_1f_cc_2_smd.csv
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
      labelMap: C:\Users\njessurun\Desktop\datagen\all_labels.csv
      valPct: 0.15
      testPct: 0.01 # Lots of unannotated images on which to test
      replace: false
      maxTestSamps: null
      maskColors: [ null ] # Don't generate overlays/rescaled masks to save time

  Semantic LinkNet:
    - s3awfs.models.linknet.LinkNetTrainingWorkflow
  Bbox LinkNet:
    - s3awfs.models.linknet.LinkNetTrainingWorkflow:
        convertMasksToBbox: True

top:
  Data Generation:
    - Create Directories
    - datagen
  Semantic LinkNet Training:
    - Create Directories
    - datagen:
      disabled: True
    - Semantic LinkNet
  Bbox LinkNet Training:
    - Create Directories
    - datagen:
      disabled: True
    - Bbox LinkNet
modules:
- s3awfs.utils
