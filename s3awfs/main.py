from __future__ import annotations
import typing as t
from functools import wraps

from pathlib import Path

from s3a.parameditors.algcollection import AlgCollection
from s3awfs.utils import WorkflowEditor
from utilitys import fns
from utilitys.typeoverloads import FilePath

import s3awfs
from s3awfs import NestedWorkflow, Workflow_T, constants, wfModules


class MainWorkflow(NestedWorkflow):
    name = 'Main Workflow'

    @fns.dynamicDocstring(_allWorkflows=[m.split('.')[-1] for m in wfModules])
    def __init__(
        self,
        folder,
        stages: list[str | t.Type[Workflow_T]],
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
        super().__init__(name=name, folder=folder)

        if multiprocess:
            constants.DEBUG = False

        useClasses = []
        if isinstance(stages, str):
            stages = [stages]
        for ii, cls in enumerate(stages):
            if isinstance(cls, str):
                cls = self.resolvePartialWorkflowNames(cls)
            useClasses.append(cls)

        for stageClass in useClasses:
            self.addWorkflow(stageClass, **kwargs)

        # Defer until here since now stages will exist
        if reset:
            self.resetRegisteredPaths()
        if createDirs:
            self.createDirs()

    @classmethod
    def fromConfig(
        cls,
        config: dict | FilePath = None,
        folder=None,
        workflow: str=None,
        run=False,
        writeConfig=None,
        **kwargs
    ):
        """
        Creates a workflow based on a configuration that likely came from a previous run's call to `saveState`.

        Note that if `config` is a dict instead of a file, the output folder must be specified.
        If ``run`` is *True*, the workflow is run with the kwargs before being returned.

        :param config: Nested dict from which to initialize. Can come from ``MainWorkflow.saveStringifiedState``
        :param folder: Where to set up this workflow. If no output folder is specified, it will be set to the
          parent folder of the config file.
        :param workflow: Which top-level workflow from the config to run
        :param run: If *True*, the config will be run after being initialized
        :param writeConfig: If *True*, the config actually consumed by this workflow will be saved over the
          specified config file. This can be useful to ensure extra kwargs passed in get uploaded to the saved config.
          If *None*, config will only be written if the the destination is different from the source. I.e. if a config
          outside the workflow folder is used, it will be saved inside the workflow folder. Otherwise, the config
          won't be overwritten.
        :param kwargs: Additional kwargs either passed to ``MainWorkflow.__init__`` or ``MainWorkflow.run`` based
          on their names. See MainWorkflow.splitInitAndRunKwargs
        """
        if (
            config is None and folder is None
            or (not isinstance(config, FilePath.__args__) and folder is None)
        ):
            raise ValueError('Either "config" or "folder" must be a filepath')
        if folder is None:
            folder = Path(config).parent
        if config is None:
            config = folder/'config.alg'
        editor = AlgCollection(
            procType=NestedWorkflow,
            procEditorType=WorkflowEditor
        ).createProcessorEditor(str(folder))
        if not isinstance(config, FilePath.__args__):
            fullConfig = config
            config = 'config.alg'
        else:
            fullConfig = fns.attemptFileLoad(config)
        if 'top' not in fullConfig:
            # Classic style
            return cls.fromClassicConfig(config, folder, run, writeConfig, **kwargs)
        if workflow is not None:
            fullConfig['active'] = workflow
        editor.loadParamValues(config, fullConfig)
        proc: NestedWorkflow = editor.curProcessor.processor
        saveName = Path(config).resolve()
        if kwargs:
            proc.updateInput(**kwargs)

        if writeConfig is None:
            writeConfig = saveName.parent != proc.workflowDir
        if writeConfig:
            editor.saveParamValues(proc.workflowDir / saveName.name)

        if run:
            proc.run()

    @classmethod
    @wraps(fromConfig)
    def fromClassicConfig(
      cls,
      config: dict | FilePath = None,
      folder=None,
      run=False,
      writeConfig=None,
      **kwargs
    ):
        if config is None and folder is None:
            raise ValueError('"config" and "folder" cannot both be *None*')
        if config is None:
            config = []
        configFile = None
        if isinstance(config, FilePath.__args__):
            configFile = Path(config).resolve()
            if folder is None:
                folder = configFile.parent
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
        initKwargs, runKwargs = cls.splitInitAndRunKwargs(kwargs)
        
        folder = Path(folder or '')
        mwf = cls(folder, **initKwargs)
        mwf.updateInput(**useConfig)
        if writeConfig is None:
            writeConfig = configFile != (folder / 'config.yml').resolve()
        if writeConfig:
            mwf.saveStringifiedConfig(**initKwargs)
        if run:
            mwf.run(**runKwargs, parent=mwf)

        return mwf

    @classmethod
    def resolvePartialWorkflowNames(cls, stageSpec: str | list[str]):
        if isinstance(stageSpec, (str, Workflow_T)):
            stageSpec = [stageSpec]
            returnSingle = True
        else:
            returnSingle = False
        nameFormatter = lambda stage: ''.join(stage.split()).lower()
        ret = []
        checked = []
        for stageName in stageSpec:
            stageName = nameFormatter(stageName)
            # Lazily load modules instead of loading all to avoid importing unnecessary long packages
            # Otherwise, a list comprehension could be used
            match = wfClass = None
            for maybeMatchModule in wfModules:
                wfClass, matchName = s3awfs.getWorkflow(maybeMatchModule, returnName=True)
                matchName = nameFormatter(matchName)
                if stageName in matchName:
                    match = wfClass
                    break
                checked.append(matchName)
            if match is None:
                # KeyError preserves '\n' in error messages due to using repr instead of str
                # Hacky workaround to accomodate this
                class KeyErrMsg(str):
                    def __repr__(self): return str(self)
                # 'checked' will contain all possible values
                msg = f'Stage "{stageName}" not recognized, must resemble one of:\n' \
                      f'{", ".join(list(checked))}'
                raise KeyError(KeyErrMsg(msg))
            else:
                ret.append(wfClass)
        if returnSingle:
            return ret[0]
        return ret

class WorkflowProcClctn(AlgCollection):
    pass
