

""" Learned Litho Model
"""


import torch.nn as nn
import torch
from utils.general_utils import conv2d
from net.simple_conv import NeuralPointwiseNet, NeuralAreawiseNet
from config import *
from net.fno import FNO2d
from param.param_fwd_litho import litho_param

def model_selector(model_choice):
    if model_choice == 'physics':
        model = LearnedLithoParamteriziedPhysics(
            hatching_distance=litho_param['hatching_distance'], sigmao=0.2, sigmac=2.5, thresh=0.5, kmax=1.0, kmin=0.5).to(device)
    elif model_choice == 'pbl3d':
        model = LearnedLitho3D(hatching_distance=litho_param['hatching_distance'], sigmao=0.2, sigmac=2.5).to(device)
    elif model_choice == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=12).to(device)
    else:
        raise NotImplementedError
    return model


class BaseLithoModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def get_aerial_image(self, masks):
        aerial_image = ...
        return aerial_image

    def get_resist_image(self, aerial_image):
        resist_image = ...
        return resist_image
    
    def forward(self, masks):
        
        aerial_image = self.get_aerial_image(masks)
        resist_image = self.get_resist_image(aerial_image)
        return resist_image

         
class LearnedLithoParamteriziedPhysics(BaseLithoModel):
    """ 3D litho model fitting from the pure physics based model.
    """
    def __init__(self, hatching_distance=0.1, sigmao=0.2, sigmac=1.5063, thresh=1.6033, kmax=1.0, kmin=0.5, alpha=5.2514, height_bias=-0.1374, sigmac_range = 2.5) -> None:
        super().__init__()

        self.hatching_distance = hatching_distance
        self.sigmao = sigmao 
        self.sigmac_range = sigmac_range
        self.kmax = kmax

        self.sigmac_param = nn.parameter.Parameter(torch.tensor(sigmac))
        self.thresh = nn.parameter.Parameter(torch.tensor([thresh]))
        self.alpha = nn.parameter.Parameter(torch.tensor([alpha]))
        self.kmin = nn.parameter.Parameter(torch.tensor([kmin]))
        self.height_bias = nn.parameter.Parameter(torch.tensor([height_bias]))

    def shrinkage_transform(self, input):
        """ 
        """
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
        # TODO illustrate what is 101
        stepxy = min(
            int(torch.div(sigma, self.hatching_distance, rounding_mode='trunc'))*6+1, 101)
        rangexy = torch.div(stepxy, 2, rounding_mode='trunc')
        xycord = torch.linspace(-rangexy, rangexy,
                                steps=stepxy).to(device)*self.hatching_distance
        kernel = torch.exp(-(xycord[:, None]**2 +
                             xycord[None, :]**2)/2/sigma**2)
        
        # norm the kernel
        kernel = kernel / torch.sum(kernel)

        return kernel[None, None]

    def get_aerial_image(self, masks):
        """ in the TPL system we experiment with the illumination is incoherent pointwise scanning. thus we model it as a 2D convolution.
        """
        illum_kernel = self.create_gaussian_kernel(self.sigmao)
        aerial_image = conv2d(masks, illum_kernel, intensity_output=True)

        return aerial_image

    def get_resist_image(self, aerial_image):
        #  Thresholding
        sigmac = torch.sigmoid(self.sigmac_param)*self.sigmac_range
        diffusion_kernel = self.create_gaussian_kernel(sigmac)
        exposure = self.threshold_approx(aerial_image)

        # Diffusion
        diffusion = conv2d(exposure, diffusion_kernel, intensity_output=True)
        
        # Shrinkage
        shrinkage = self.shrinkage_transform(diffusion)
        resist_image = (exposure*shrinkage+self.height_bias)
        return resist_image

    def forward(self, masks):
    
        aerial_image = self.get_aerial_image(masks)
        resist_image = self.get_resist_image(aerial_image)

        return resist_image


class LearnedLitho3D(BaseLithoModel):
    """ 3D litho model fitting from the physics+NN based model.
    """
    def __init__(self, hatching_distance=0.1, sigmao=0.2, sigmac=0.25) -> None:
        super().__init__()
        self.hatching_distance = hatching_distance # lateral hatching distance of the litho system.

        self.sigmao = sigmao
        self.sigmac_range = sigmac
        self.sigmac_param = nn.parameter.Parameter(torch.tensor(sigmac))

        self.thresh_approx = NeuralPointwiseNet()
        self.shrink_approx = NeuralPointwiseNet()
        self.out_layer = NeuralAreawiseNet()

    def create_gaussian_kernel(self, sigma):
        """Gaussian kernel for the illumination"""
        stepxy = min(
            int(torch.div(sigma, self.hatching_distance, rounding_mode='trunc'))*6+1, 101)
        rangexy = torch.div(stepxy, 2, rounding_mode='trunc')
        xycord = torch.linspace(-rangexy, rangexy,
                                steps=stepxy).to(device)*self.hatching_distance
        kernel = torch.exp(-(xycord[:, None]**2 +
                             xycord[None, :]**2)/2/sigma**2)
        kernel = kernel / torch.sum(kernel)
        return kernel[None, None]
    
    def get_aerial_image(self, masks):
        """ in the TPL system we experiment with the illumination is incoherent pointwise scanning. thus we model it as a 2D convolution.
        """
        illum_kernel = self.create_gaussian_kernel(self.sigmao)
        aerial_image = conv2d(masks, illum_kernel, intensity_output=True)
        
        return aerial_image
    
    def get_resist_image(self, aerial_image):
        
        # Thresholding
        exposure = self.thresh_approx(
        aerial_image)
        
        # Diffusion
        sigmac = torch.sigmoid(self.sigmac_param)*self.sigmac_range
        diffusion_kernel = self.create_gaussian_kernel(sigmac)
        diffusion = conv2d(exposure, diffusion_kernel, intensity_output=True)
        
        # Shrinkage
        shrinkage = self.shrink_approx(diffusion)
        resist_image = self.out_layer(exposure*shrinkage)
        return resist_image
        
    def forward(self, masks):
        
        aerial_image = self.get_aerial_image(masks)
        resist_image = self.get_resist_image(aerial_image)
        
        return resist_image
