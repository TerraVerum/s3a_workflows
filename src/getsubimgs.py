from pathlib import Path

from s3a import ProjectData

PROJ_FILE = Path.home()/'Desktop/proj-smd/proj-smd.s3aprj'
EXPORT_DIR = Path('/home/ntjess/Documents/mini-smd-export/')

prj = ProjectData(PROJ_FILE)
prj.exportAnnotations(
  EXPORT_DIR,
  'comp.imgs.zip',
  combine=True,
  includeImages=False,
  resizeOpts=dict(
    shape=(300, int(300*1.75)),
    keepAspectRatio=True,
    allowReorient=True
  ),
  labelField='Designator'
)
