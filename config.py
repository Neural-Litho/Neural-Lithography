

import torch
import torch.nn as nn
import torch.nn.functional as F

gpu_id = 0
gpu_name = 'cuda:{}'.format(gpu_id)
device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
