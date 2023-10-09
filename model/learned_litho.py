

import torch.nn as nn
import torch
from utils.general_utils import conv2d
from net.nff_network import NeuralPointwiseNet, NeuralAreawiseNet
from config import *


class LearnedLitho3D(nn.Module):

    def __init__(self, dx=0.1, sigmao=0.2, sigmac=1.5063, thresh=1.6033, kmax=1.0, kmin=0.5, alpha=5.2514, height_bias=-0.1374) -> None:
        super().__init__()
        
        self.dx = dx
        self.sigmao = sigmao
        self.sigmac_range = 2.5
        self.sigmac_param = nn.parameter.Parameter(torch.tensor(sigmac))

        self.thresh = nn.parameter.Parameter(torch.tensor([thresh]))
        self.alpha = nn.parameter.Parameter(torch.tensor([alpha]))
        self.kmax = kmax
        self.kmin = nn.parameter.Parameter(torch.tensor([kmin])) 
        self.height_bias = nn.parameter.Parameter(torch.tensor([height_bias])) 

    def shrinkage_transform(self, input):
        kmin = torch.sigmoid(self.kmin)
        dmax = torch.amax(input, dim=(2, 3), keepdim=True)
        sh = (self.kmax-kmin)/dmax*input+kmin
        return sh
    
    def threshold_approx(self, aerial):
        eta = torch.sigmoid(self.thresh)
        deno = torch.tanh(self.alpha*eta) + \
            torch.tanh(self.alpha*(1-eta))
        output = (torch.tanh(self.alpha*eta) +
                      torch.tanh(self.alpha*(aerial-eta)))/deno
        return output
            
    def create_gaussian_kernel(self, sigma):
        stepxy = min(
            int(torch.div(sigma, self.dx, rounding_mode='trunc'))*6+1, 101)
        rangexy = torch.div(stepxy, 2, rounding_mode='trunc')
        xycord = torch.linspace(-rangexy, rangexy,
                                steps=stepxy).to(device)*self.dx
        kernel = torch.exp(-(xycord[:, None]**2 +
                             xycord[None, :]**2)/2/sigma**2)
        kernel = kernel / torch.sum(kernel)
        return kernel[None, None]
    
    def forward(self, masks):
        masks = masks/100

        illum_kernel = self.create_gaussian_kernel(self.sigmao)
        sigmac = torch.sigmoid(self.sigmac_param)*self.sigmac_range
        diffusion_kernel = self.create_gaussian_kernel(sigmac)
        
        exposure = conv2d(masks, illum_kernel, intensity_output=True)
        exposure = self.threshold_approx(exposure)
        
        diffusion = conv2d(exposure, diffusion_kernel, intensity_output=True)
        shrinkage = self.shrinkage_transform(diffusion)
        out = (exposure*shrinkage+self.height_bias)*100

        return out


class LearnedLitho3D2(nn.Module):

    def __init__(self, dx=0.1, sigmao=0.2, sigmac=2.5) -> None:
        super().__init__()
        self.dx = dx
        
        self.sigmao = sigmao 
        self.sigmac_range = sigmac
        self.sigmac_param = nn.parameter.Parameter(torch.tensor(sigmac))
        
        self.thresh_approx = NeuralPointwiseNet()
        self.shrink_approx = NeuralPointwiseNet()
        self.out_layer = NeuralAreawiseNet()
    
    def create_gaussian_kernel(self, sigma):
        stepxy = min(int(torch.div(sigma, self.dx, rounding_mode='trunc'))*6+1, 101)
        rangexy = torch.div(stepxy, 2, rounding_mode='trunc')
        xycord = torch.linspace(-rangexy, rangexy, steps=stepxy).to(device)*self.dx
        kernel = torch.exp(-(xycord[:, None]**2 + xycord[None, :]**2)/2/sigma**2)
        kernel = kernel / torch.sum(kernel)
        return kernel[None, None]   
    
    def forward(self, masks):
        masks = masks/100
        illum_kernel = self.create_gaussian_kernel(self.sigmao)
        sigmac = torch.sigmoid(self.sigmac_param)*self.sigmac_range
        diffusion_kernel = self.create_gaussian_kernel(sigmac)        
       
        exposure = self.thresh_approx(conv2d(masks, illum_kernel, intensity_output=True))
        
        diffusion = conv2d(exposure, diffusion_kernel, intensity_output=True)
        shrinkage = self.shrink_approx(diffusion)
       
        out = self.out_layer(exposure*shrinkage)*100
        
        return out
