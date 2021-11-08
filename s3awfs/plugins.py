from pathlib import Path

from s3a.parameditors.algcollection import AlgCollection, AlgParamEditor
from utilitys import fns

from s3awfs.utils import NestedWorkflow

def pathCtor(constructor, node):
  return Path(constructor.construct_scalar(node))
fns.loader.constructor.add_constructor('!Path', pathCtor)

class WorkflowEditor(AlgParamEditor):
  def _resolveProccessor(self, proc):
    retProc = super()._resolveProccessor(proc)
    if retProc is not None:
      # Only one top processor can exist, so setting the workflow dir on subfolders
      # will keep each primitive proc at the saveDir level
      retProc.localFolder = self.saveDir
    return retProc

def ALL_PLUGINS():
  pass