import logging
import random

import numpy as np
import torch

try:
    import psutil
except Exception:
    psutil = None

logger = logging.getLogger(__name__)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

process = psutil.Process() if psutil is not None else None
SAVE_DIR = None
JARVIS_LOADER_CACHE = {}
