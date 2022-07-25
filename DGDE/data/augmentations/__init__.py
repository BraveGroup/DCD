import logging
import numpy as np 

from .augmentations import (
    RandomHorizontallyFlip,
    RandomResize,
    Compose,
)

from config import cfg

aug_list = [RandomHorizontallyFlip,RandomResize]

logger = logging.getLogger("DGDE.augmentations")

def get_composed_augmentations(aug_params=None):
    if aug_params is None:
        aug_params = cfg.INPUT.AUG_PARAMS
    augmentations = []
    for i, (aug, aug_param) in enumerate(zip(aug_list, aug_params)):
        if (i==0 and aug_param[0] > 0) or i==1:
            if len(aug_param)==1:
                augmentations.append(aug(*aug_param))
            elif len(aug_param)==2:
                augmentations.append(aug(aug_param[0],aug_param[1]))
            else:
                raise ValueError('len of aug_param is illegal')

            logger.info("Using {} aug with params {}".format(aug, aug_param))

    return Compose(augmentations)
