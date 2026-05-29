import torch
import torch.nn as nn
import numpy as np
import random
import time
from torch.backends import cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000000) % (2 ** 32)
        print(f"Using random seed: {seed}")
    else:
        print(f"Using fixed seed: {seed}")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True