import sys

from qtextras import fns
from s3a import mkQApp

from s3awfs import allWorkflows
from s3awfs.main import MainWorkflow
from s3awfs.utils import argparseHelpAction


def main_cli():
    mkQApp()
    parser = fns.makeCli(
        MainWorkflow.fromConfig, parserKwargs=dict(add_help=False), run=True
    )
    # Expensive, avoid unless requested
    if "--help" in sys.argv:
        parser.register(
            "action",
            "help",
            argparseHelpAction(
                MainWorkflow("garbageasdf", allWorkflows(), createDirectories=False)
            ),
        )
        parser.add_argument("--help", action="help")
    kwargs = vars(parser.parse_args())
    MainWorkflow.fromConfig(**kwargs)


if __name__ == "__main__":
    main_cli()
