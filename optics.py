

from config import *
import math
from propagator import RSCProp
from utils.gumbel_max_pytorch import gumbel_softmax


class HoloFwd(nn.Module):
    def __init__(self, input_dx, input_shape, output_dx, output_shape, wave_lengths, z, pad_scale, Delta_n=0.545) -> None:
        super().__init__()
        
        self.propagator = RSCProp(
            input_dx, input_shape, output_dx, output_shape, 
            wave_lengths, z, pad_scale)
        self.transfer_factor = 2*math.pi/wave_lengths*Delta_n
        
    def forward(self, input):
        
        # transfer height unit nm to phase 
        phase = input/10*self.transfer_factor
        x = torch.exp(1j*phase)
       
        # norm source input 
        x = x/ torch.sqrt(torch.prod(torch.tensor(x.shape[-2:])))
        x = self.propagator(x)
        return x 

class DOE(nn.Module):
    """Some InformatMyModule"""

    def __init__(self, num_partition, doe_level, output_size, doe_layers=1, doe_type='1d') -> None:
        super(DOE, self).__init__()
        self.doe_type = doe_type
        self.doe_size = num_partition
        if self.doe_type == '2d':
            self.logits = nn.parameter.Parameter(
                torch.rand(doe_layers, num_partition, num_partition, doe_level), requires_grad=True)
        elif self.doe_type == '1d':
            self.logits = nn.parameter.Parameter(
                torch.rand(doe_layers, num_partition, doe_level), requires_grad=True)
            self.generate_mesh_mapping()
        self.doe_level = doe_level
        self.doe_layers = doe_layers
        self.level_logits = torch.arange(0, self.doe_level).to(device)
        self.m = nn.Upsample(scale_factor=output_size[0]/
                             num_partition, mode='nearest')

    def logits_to_doe_profile(self):
        _, doe_res = self.logits.max(dim=-1)
        if self.doe_type == '1d':
            doe_images = torch.zeros(
                [self.doe_layers, self.doe_size, self.doe_size])
            for i in range(self.doe_layers):
                doe_images[i] = doe_res[i, self.inds]
        elif self.doe_type == '2d':
            doe_images = doe_res
        return doe_images

    def generate_mesh_mapping(self):
        # self.inds contains a 2D tensor of index point to 1D doe values
        size = self.doe_size
        coord_x = torch.linspace(-size/2, size/2, int(size))
        coord_y = torch.linspace(-size/2, size/2, int(size))
        meshx, meshy = torch.meshgrid(coord_x, coord_y, indexing='ij')
        meshrho = torch.sqrt(meshx ** 2 + meshy ** 2)
        rho = math.sqrt(2)*(torch.arange(0, size // 2, dtype=torch.double))
        distance = torch.abs(meshrho[:, :, None] - rho[None, None, :])
        self.inds = torch.argmin(distance, dim=2)

    def get_doe_sample(self):
      # Sample soft categorical using reparametrization trick:
        sample_one_hot = gumbel_softmax(self.logits, tau=1, hard=False).to(device)
       
        if self.doe_type == '2d':
            doe_sample = (sample_one_hot *
                          self.level_logits[None, None, None, :]).sum(dim=-1)
        elif self.doe_type == '1d':
            doe_sample_1d = (sample_one_hot *
                             self.level_logits[None, None, :]).sum(dim=-1)
            doe_sample = torch.zeros(
                [self.doe_layers, self.doe_size, self.doe_size])
            for i in range(self.doe_layers):
                doe_sample[i] = doe_sample_1d[i, self.inds]

        doe_sample = self.m(doe_sample[None, :, :, :])
        
        if torch.isnan(doe_sample).any():
            raise
        
        return doe_sample.to(device)
