from __future__ import annotations

from pathlib import Path
import sys

# Some systems need qt initialized before cv gets imported
import pyqtgraph as pg
pg.mkQApp()

from s3awfs.generic import MainWorkflow
from s3awfs.workflows import allWorkflows
from s3awfs.workflows.constants import DEFAULT_RESIZE_OPTS, SMD_INIT_OPTS
from s3awfs.workflows.utils import argparseHelpAction


from utilitys import fns

def main_rgbFeatures512(**kwargs):
    kwargs: dict = {
        'resizeOpts': DEFAULT_RESIZE_OPTS,
        **SMD_INIT_OPTS,
        **kwargs,
    }
    folder = kwargs['folder'] = Path(kwargs.pop('folder', None) or Path.home()/'Desktop/rgb_features_512')
    kwargs['labelMapFile'] = kwargs.get('labelMapFile') or folder/'aliased_labels.csv'
    init, run = MainWorkflow.splitInitAndRunKwargs(kwargs)
    mwf = MainWorkflow(**init)
    return mwf.run(**run)

def main_cli():
    parser = fns.makeCli(main_rgbFeatures512, parserKwargs=dict(add_help=False))
    # Expensive, avoid unless requested
    if '--help' in sys.argv:
        parser.register('action', 'help', argparseHelpAction(MainWorkflow('garbageasdf', allWorkflows(), createDirs=False)))
        parser.add_argument('--help', action='help')
    kwargs = vars(parser.parse_args())
    if 'config' in kwargs:
        MainWorkflow.runFromConfig(**kwargs)
    else:
        main_rgbFeatures512(**kwargs)

if __name__ == '__main__':
    main_cli()
