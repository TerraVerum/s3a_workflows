from __future__ import annotations

from pathlib import Path

from utilitys import fns

from s3awfs.main import MainWorkflow
from s3awfs.workflows import *
from s3awfs.workflows.constants import SMD_INIT_OPTS

OUTPUT_PARENT = Path.home()/'Desktop'

here = Path(__file__).parent


def fpicMain(config: str, outputParent=OUTPUT_PARENT, **kwargs):
    if not config.endswith('.yml'):
        config += '.yml'
    fullConfig = here/config
    kwargs.setdefault('writeConfig', True)
    kwargs.setdefault('folder', Path(outputParent)/fullConfig.stem)
    if 'linknet' in config:
        runLinknet(fullConfig, **kwargs)
    else:
        kwargs.update(SMD_INIT_OPTS)
        MainWorkflow.fromConfig(fullConfig, run=True, **kwargs)


def runLinknet(fullConfig: str | Path, **kwargs):
    mwf = MainWorkflow.fromConfig(fullConfig, **kwargs)
    # Need to point datas to the right folders
    for wfType in FormattedInputWorkflow, ComponentImagesWorkflow, PngExportWorkflow, TrainValidateTestSplitWorkflow:
        curWf: WorkflowDir = mwf.get(wfType)
        curWf.workflowDir = OUTPUT_PARENT/'datagen'
    mwf.run(parent=mwf)

def fpicMain_cli():
    parser = fns.makeCli(fpicMain)
    fpicMain(**vars(parser.parse_args()))

if __name__ == '__main__':
    fpicMain_cli()