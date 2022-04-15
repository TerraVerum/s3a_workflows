import numpy as np
from s3a.constants import PRJ_ENUMS

DEFAULT_RESIZE_OPTS = dict(shape=(512, 512), rotationDeg=PRJ_ENUMS.ROT_OPTIMAL)

# Determines whether multiprocessing uses multiple cores or is disabled
DEBUG = True

DEFAULT_RGB_CMAP = "PAL-relaxed_bright"

RNG = np.random.default_rng()
