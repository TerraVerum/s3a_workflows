from __future__ import annotations

import os
from pathlib import Path

from s3a.parameditors.algcollection import AlgCollection
from s3awfs import NestedWorkflow
from s3awfs.utils import WorkflowEditor
from utilitys import fns

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

here = Path(__file__).parent
OUTPUT_PARENT = Path('/shared/fpic-paper-experiments/')
if not OUTPUT_PARENT.exists():
    OUTPUT_PARENT = Path.home()/'Desktop'

def fpicMain(workflow: str, outputParent=OUTPUT_PARENT, **kwargs):
    fullConfig = here / 'config.alg'
    editor = AlgCollection(procType=NestedWorkflow, procEditorType=WorkflowEditor).createProcessorEditor(str(outputParent))
    editor.loadParamValues(fullConfig)
    editor.changeActiveProcessor(workflow)
    proc: NestedWorkflow = editor.curProcessor.processor
    proc.saveStringifiedConfig()
    if kwargs:
        proc.updateInput(**kwargs)
    proc.run()

def fpicMain_cli():
    parser = fns.makeCli(fpicMain)
    fpicMain(**vars(parser.parse_args()))

if __name__ == '__main__':
    fpicMain_cli()
