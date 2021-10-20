from __future__ import annotations

import inspect
import typing as t
from pathlib import Path

from utilitys import fns, AtomicProcess
from utilitys.typeoverloads import FilePath

import workflows.constants
from workflows import NestedWorkflow, Workflow_T, allWorkflows, WorkflowDir
from workflows.utils import stringifyDict

_allWorkflows = allWorkflows()

class MainWorkflow(NestedWorkflow):
    name = 'Main Workflow'

    @fns.dynamicDocstring(_allWorkflows=list(_allWorkflows))
    def __init__(
        self,
        folder,
        stages: list[str | Workflow_T]=None,
        multiprocess=False,
        createDirs=True,
        reset=False,
        **kwargs
    ):
        """
        Performs workflows as prescribed by every stage defined in ``stages``
        :param folder: Where to generate outputs, subfolders, etc.
        :param stages: Which stages to run. Defaults to {_allWorkflows}
        :param multiprocess: If *True*, each stage will be run with multiprocessing enabled. Warning -- this can have
          tremendous CPU and memory consumption
        :param createDirs: If *True*, subdirectories will be created before running the workflow
        :param reset: If *True*, workflow stages identified by ``stages`` will be reset (all subdirectories deleted
          and files removed) before running
        :param kwargs: Additional keywods for workflow creation
        """
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

    @classmethod
    def resolvePartialWorkflowNames(cls, stageSpec: str | list[str]):
        if isinstance(stageSpec, (str, Workflow_T)):
            stageSpec = [stageSpec]
            returnSingle = True
        else:
            returnSingle = False
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
    def runFromConfig(cls, config: dict|FilePath=None, folder=None, **kwargs):
        """
        Runs a workflow based on a configuration that likely came from a previous run's call to `saveState`.
        If no output folder is specified, it will be set to the parent folder of the config file.
        Note that if `config` is a dict instead of a file, the output folder must be specified.
        """
        if config is None and folder is None:
            raise ValueError('"config" and "folder" cannot both be *None*')
        if config is None:
            config = []
        if isinstance(config, FilePath.__args__):
            if folder is None:
                folder = Path(config).parent
            config = fns.attemptFileLoad(config)
        if cls.name in config:
            # Pop the top level if given
            config = config[cls.name]

        # Remove prefix stage number if it exists
        nameFmt = lambda name: name.split('. ', maxsplit=1)[-1]
        useConfig = {}
        for stage in config:
            if isinstance(stage, dict):
                for stageK, stageV in stage.items():
                    useConfig[nameFmt(stageK)] = stageV
        if 'Initialization' in useConfig:
            # Came from a previously saved run
            kwargs = {**useConfig.pop('Initialization'), **kwargs}

        useNames = [nameFmt(name) for name in useConfig]
        kwargs.setdefault('stages', useNames)
        init, run = MainWorkflow.splitInitAndRunKwargs(kwargs)

        mwf = MainWorkflow(folder, **init)
        mwf.updateInput(**useConfig, graceful=True)
        state = mwf.saveStringifiedConfig(**init)
        mwf.run(**run)

        return state

    def saveStringifiedConfig(self, **initKwargs):
        state = self.saveState(includeDefaults=True)
        # Make a dummy process for input parameters just to easily save its state
        initState = AtomicProcess(self.__init__, name='Initialization', interactive=False, **initKwargs).saveState(includeDefaults=True)
        state[self.name].insert(0, initState)

        # Some values are unrepresentable in their natural form (e.g. Paths)
        state = stringifyDict(state)
        fns.saveToFile(state, self.workflowDir / 'config.yml')
        return state

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
