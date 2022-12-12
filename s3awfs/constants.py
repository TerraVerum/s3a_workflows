import numpy as np
from s3a.constants import PRJ_ENUMS

DEFAULT_RESIZE_OPTS = dict(shape=(512, 512), rotationDegrees=PRJ_ENUMS.ROTATION_OPTIMAL)

# Determines whether multiprocessing uses multiple cores or is disabled
DEBUG = True

DEFAULT_RGB_CMAP = "plasma"

DUMMY_FGND_VALUE = "FGND"

RNG = np.random.default_rng()
