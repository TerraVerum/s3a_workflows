import pydoc

from .utils import NestedWorkflow, Workflow_T, WorkflowDirectory

__version__ = "0.2.0"

# It would be easy enough to automate this, but it is better to ensure proper processing order

wfModules = [
    "fmtinput.FormattedInputWorkflow",
    "compimgs.ComponentImagesWorkflow",
    "regionpropfeats.RegionPropertiesWorkflow",
    "png.PngExportWorkflow",
    "tvtsplit.TrainValidateTestSplitWorkflow",
    "imagefeats.FeatureTransformerWorkflow",
    "models.tensorflow.TensorflowTrainingWorkflow",
    "models.linknet.LinkNetTrainingWorkflow",
]


def allWorkflows():
    ret = {}
    for wf in wfModules:
        cls, name = getWorkflow(wf, returnName=True)
        ret[name] = cls
    return ret


def getWorkflow(moduleStr, returnName=False):
    # False positive
    # noinspection PyTypeChecker
    wfCls: WorkflowDirectory = pydoc.locate(f"{__name__}.{moduleStr}")
    name = wfCls.name or moduleStr.split(".")[-1]
    if returnName:
        return wfCls, name
    return wfCls


from .main import MainWorkflow
