import os
from datetime import datetime
import logging
import random
import numpy as np
import pdb
import torch


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    logger = logging.getLogger('DGDE.seed')

    if seed is None:
        seed = (
                os.getpid()
                # + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
        )
        logger.info("Using a generated random seed {}".format(seed))
    else:
        pass
        # logger.info("Using a specified random seed {}".format(seed))
    
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
