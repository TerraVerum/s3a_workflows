import numpy as np
from autobom.constants import TEMPLATES_DIR
from s3a.constants import PRJ_ENUMS

SMD_INIT_OPTS = dict(
    labelField='Designator',
    s3aProj=TEMPLATES_DIR/'proj_smd.s3aprj',
)

DEFAULT_RESIZE_OPTS = dict(
    shape=(512, 512),
    rotationDeg=PRJ_ENUMS.ROT_OPTIMAL
)

# Determines whether multiprocessing uses multiple cores or is disabled
DEBUG = True

RNG = np.random.default_rng()