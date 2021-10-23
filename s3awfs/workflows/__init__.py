from .compimgs import ComponentImagesWorkflow
from .fmtinput import FormattedInputWorkflow
from .models.linknet import LinkNetTrainingWorkflow
from .png import PngExportWorkflow
from .regionpropfeats import RegionPropertiesWorkflow
from .imagefeats import FeatureTransformerWorkflow
from .tvtsplit import TrainValidateTestSplitWorkflow
from .utils import WorkflowDir, NestedWorkflow

# It would be easy enough to automate this, but it is better to ensure proper processing order
__all__  = [
    'FormattedInputWorkflow',
    'ComponentImagesWorkflow',
    'RegionPropertiesWorkflow',
    'PngExportWorkflow',
    'TrainValidateTestSplitWorkflow',
    'FeatureTransformerWorkflow',
    'LinkNetTrainingWorkflow',
    'WorkflowDir',
    'NestedWorkflow',
    'allWorkflows',
    'Workflow_T'
]

Workflow_T = (WorkflowDir, NestedWorkflow)

def allWorkflows():
    import sys, inspect
    ret = {}
    module = sys.modules[__name__]
    for name in __all__:
        cls = getattr(module, name)
        if inspect.isclass(cls) and issubclass(cls, Workflow_T) and cls not in Workflow_T:
            ret[cls.name or name] = cls
    return ret
