

from cuda_config import *
import math
from task_optics.propagator import RSCProp


class FreeSpaceFwd(nn.Module):
    def __init__(self, input_dx, input_shape, output_dx, output_shape, wave_lengths, z, pad_scale, Delta_n) -> None:
        super().__init__()
        
        self.propagator = RSCProp(
            input_dx, input_shape, output_dx, output_shape, 
            wave_lengths, z, pad_scale)
        self.transfer_factor = 2 * math.pi / wave_lengths * Delta_n
        
    def forward(self, input):
        # transfer height unit nm to phase 
        phase = input * self.transfer_factor
        x = torch.exp(1j * phase)
       
        # norm source input 
        x = x/ torch.sqrt(torch.prod(torch.tensor(x.shape[-2:])))
        x = self.propagator(x)
        return x 

