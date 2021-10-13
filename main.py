from __future__ import annotations

import inspect
import sys
import typing as t
from pathlib import Path

# Some systems need qt initialized before cv gets imported
import pyqtgraph as pg
from utilitys.typeoverloads import FilePath

pg.mkQApp()

from utilitys import fns, ProcessIO
from workflows.constants import SMD_INIT_OPTS, DEFAULT_RESIZE_OPTS
import workflows.constants
from workflows import allWorkflows, Workflow_T
from workflows.utils import NestedWorkflow, WorkflowDir, stringifyDict, argparseHelpAction

class MainWorkflow(NestedWorkflow):
    name = 'Main Workflow'
    def __init__(
        self,
        folder,
        stages: list[str | Workflow_T]=None,
        multiprocess=False,
        createDirs=True,
        reset=False,
        **kwargs
    ):
        name = kwargs.pop('name', self.name)
        super().__init__(folder, name=name)

        if multiprocess:
            workflows.constants.DEBUG = False

        defaultClasses = allWorkflows()
        if stages is None:
            # By default, use all stages except linknet
            stages = list(allWorkflows())[:-1]
        useClasses = self.resolvePartialWorkflowNames(stages)
        if not isinstance(useClasses, list):
            useClasses = [useClasses]
        for name, stageClass in defaultClasses.items():
            stage = self.addWorkflow(stageClass, **kwargs)
            if stageClass not in useClasses:
                stage.disabled = True

        # Defer until here since now stages will exist
        if reset:
            self.resetRegisteredPaths()
        if createDirs:
            self.createDirs()

    def disableStages(self, *stageClasses: t.Type[WorkflowDir]):
        for cls in stageClasses:
            self.get(cls).disabled = True

    def run(self, io: ProcessIO = None, disable=False, **runKwargs):
        # Prioritize user kwargs if there are duplicates
        runKwargs = {
            'resizeOpts': DEFAULT_RESIZE_OPTS,
            **SMD_INIT_OPTS,
            **runKwargs,
            **(io or {})
        }
        ret = super().run(disable=disable, **runKwargs)
        # Some values are unrepresentable in their natural form (e.g. Paths)
        state = stringifyDict(self.saveState(includeDefaults=True))
        fns.saveToFile(state, self.workflowDir / 'config.yml')
        return ret

    @classmethod
    def resolvePartialWorkflowNames(cls, stageSpec: str | list[str]):
        if isinstance(stageSpec, (str, Workflow_T)):
            stageSpec = [stageSpec]
            returnSingle = True
        else:
            returnSingle = False
        _allWorkflows = allWorkflows()
        wfNames = {''.join(stageName.split()).lower(): stageName for stageName in _allWorkflows}
        ret = []
        for stageName in stageSpec:
            stageName = ''.join(stageName.split()).lower()
            matches = [formatted for matchName, formatted in wfNames.items() if stageName in matchName]
            if not any(matches):
                raise KeyError(f'Stage "{stageName}" not recognized, must resemble one of:\n'
                               f'{", ".join(list(_allWorkflows))}')
            else:
                formattedName = matches[0]
                wfClass = _allWorkflows[formattedName]
            ret.append(wfClass)
        if returnSingle:
            return ret[0]
        return ret

    @classmethod
    def runFromConfig(cls, config: dict|FilePath, outputFolder=None, **kwargs):
        """
        Runs a workflow based on a configuration that likely came from a previous run's call to `saveState`.
        If no output folder is specified, it will be set to the parent folder of the config file.
        Note that if `config` is a dict instead of a file, the output folder must be specified.
        """
        if isinstance(config, FilePath.__args__):
            if outputFolder is None:
                outputFolder = Path(config).parent
            config = fns.attemptFileLoad(config)
        if 'Main Workflow' in config:
            # Pop the top level if given
            config = config['Main Workflow']
        useConfig = {}
        for stage in config:
            if isinstance(stage, dict):
                useConfig.update(stage)
        useNames = [name.split(maxsplit=1)[-1] for name in useConfig]
        kwargs.setdefault('stages', useNames)
        init, run = MainWorkflow.splitInitAndRunKwargs(kwargs)
        mwf = MainWorkflow(outputFolder, **init)
        mwf.updateInput(**useConfig, graceful=True)
        return mwf.run(**run)

    @classmethod
    def splitInitAndRunKwargs(cls, kwargs):
        """
        Converts a dict of potentially both __init__ keywords and run() keywords into two separate dicts
        """
        initSpec = set(inspect.signature(cls.__init__).parameters)
        initKwargs = {}
        for kw in initSpec:
            if kw in kwargs:
                initKwargs[kw] = kwargs.pop(kw)
        return initKwargs, kwargs


def main_rgbFeatures512(**kwargs):
    outputFolder = kwargs['outputFolder'] = Path(kwargs.pop('outputFolder', None) or Path.home()/'Desktop/rgb_features_512')
    kwargs['labelMapFile'] = kwargs.get('labelMapFile') or outputFolder/'aliased_labels.csv'
    init, run = MainWorkflow.splitInitAndRunKwargs(kwargs)
    mwf = MainWorkflow(outputFolder, **init)
    return mwf.run(**run)

def main_cli():
    parser = fns.makeCli(main_rgbFeatures512, parserKwargs=dict(add_help=False))
    kwargs = vars(parser.parse_args())
    if 'config' in kwargs:
        MainWorkflow.runFromConfig(**kwargs)
    else:
        main_rgbFeatures512(**kwargs)

if __name__ == '__main__':
    main_cli()
