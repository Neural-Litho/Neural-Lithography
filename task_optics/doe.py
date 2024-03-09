
import torch.nn as nn
import torch
from config import device
import math
from utils.gumbel_max_pytorch import gumbel_softmax

class DOE(nn.Module):
    """Sampling the paramterized DOE. If w/ litho model in the design loop, the result is not a DOE but a mask to print in the litho system.
    """

    def __init__(self, num_partition, doe_num_level, output_size, slicing_distance, doe_layers=1, doe_type='1d') -> None:
        super(DOE, self).__init__()
        self.doe_type = doe_type
        self.doe_size = num_partition
        self.slicing_distance = slicing_distance
        
        if self.doe_type == '2d':
            self.logits = nn.parameter.Parameter(
                torch.rand(doe_layers, num_partition, num_partition, doe_num_level), requires_grad=True)

        elif self.doe_type == '1d':
            self.logits = nn.parameter.Parameter(
                torch.rand(doe_layers, num_partition, doe_num_level), requires_grad=True)
            self.indices = self.generate_mesh_mapping(self.doe_size)
        self.doe_num_level = doe_num_level
        self.doe_layers = doe_layers
        self.level_logits = torch.arange(0, self.doe_num_level).to(device)
        self.m = nn.Upsample(scale_factor=output_size[0] /
                             num_partition, mode='nearest')

    def logits_to_doe_profile(self):
        _, doe_res = self.logits.max(dim=-1)
        if self.doe_type == '1d':
            doe_images = torch.zeros(
                [self.doe_layers, self.doe_size, self.doe_size])
            for i in range(self.doe_layers):
                doe_images[i] = doe_res[i, self.indices]
        elif self.doe_type == '2d':
            doe_images = doe_res
        return doe_images

    def generate_mesh_mapping(self, doe_size):
        # self.indices contains a 2D tensor of index point to 1D doe values
        coord_x = torch.linspace(-doe_size/2, doe_size/2, int(doe_size))
        coord_y = torch.linspace(-doe_size/2, doe_size/2, int(doe_size))

        meshx, meshy = torch.meshgrid(coord_x, coord_y, indexing='ij')
        meshrho = torch.sqrt(meshx ** 2 + meshy ** 2)

        rho = math.sqrt(2)*(torch.arange(0, doe_size // 2, dtype=torch.double))
        distance = torch.abs(meshrho[:, :, None] - rho[None, None, :])

        indices = torch.argmin(distance, dim=2)
        return indices

    def get_doe_sample(self):
      # Sample soft categorical using reparametrization trick:
        sample_one_hot = gumbel_softmax(
            self.logits, tau=1, hard=False).to(device)

        if self.doe_type == '2d':
            doe_sample = (sample_one_hot *
                          self.level_logits[None, None, None, :]).sum(dim=-1)
        elif self.doe_type == '1d':
            doe_sample_1d = (sample_one_hot *
                             self.level_logits[None, None, :]).sum(dim=-1)
            doe_sample = torch.zeros(
                [self.doe_layers, self.doe_size, self.doe_size])
            for i in range(self.doe_layers):
                doe_sample[i] = doe_sample_1d[i, self.indices]

        doe_sample = self.m(doe_sample[None, :, :, :])

        # convert doe sample from levels to digits
        doe_sample *= self.slicing_distance 

        if torch.isnan(doe_sample).any():
            raise

        return doe_sample.to(device)
