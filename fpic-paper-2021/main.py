from __future__ import annotations

from pathlib import Path

from s3awfs.main import MainWorkflow
from utilitys import fns

here = Path(__file__).resolve().parent
OUTPUT_PARENT = Path("/shared/fpic-paper-experiments/")
if not OUTPUT_PARENT.exists():
    OUTPUT_PARENT = Path.home() / "Desktop"


def fpicMain(workflow: str, outputParent=OUTPUT_PARENT, **kwargs):
    fullConfig = here / "config.alg"
    return MainWorkflow.fromConfig(
        fullConfig, outputParent, workflow, run=True, **kwargs
    )


def fpicMain_cli():
    parser = fns.makeCli(fpicMain)
    fpicMain(**vars(parser.parse_args()))


if __name__ == "__main__":
    fpicMain_cli()
