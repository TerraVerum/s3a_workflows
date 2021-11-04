import pydoc

from s3awfs.utils import WorkflowDir, NestedWorkflow
from .__version__ import __version__

# It would be easy enough to automate this, but it is better to ensure proper processing order

wfModules  = [
    'fmtinput.FormattedInputWorkflow',
    'compimgs.ComponentImagesWorkflow',
    'regionpropfeats.RegionPropertiesWorkflow',
    'png.PngExportWorkflow',
    'tvtsplit.TrainValidateTestSplitWorkflow',
    'imagefeats.FeatureTransformerWorkflow',
    'models.linknet.LinkNetTrainingWorkflow',
]

Workflow_T = (WorkflowDir, NestedWorkflow)
def allWorkflows():
    ret = {}
    for wf in wfModules:
        cls, name = getWorkflow(wf, returnName=True)
        ret[name] = cls
    return ret

def getWorkflow(moduleStr, returnName=False):
    # False positive
    # noinspection PyTypeChecker
    wfCls: WorkflowDir = pydoc.locate(f'{__name__}.{moduleStr}')
    name = wfCls.name or moduleStr.split('.')[-1]
    if returnName:
        return wfCls, name
    return wfClss