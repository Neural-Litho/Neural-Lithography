

import torch
import torch.nn as nn
import torch.nn.functional as F

gpu_id = 0
gpu_name = 'cuda:{}'.format(gpu_id)
device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Always use the 2nd gpu if there is more than one (Modified accordingly)
# if torch.cuda.device_count() > 1:
#     device_num = 1
# else:
#     device_num = 0

# device = torch.device("cuda:" + str(device_num) if torch.cuda.is_available() else "cpu")
