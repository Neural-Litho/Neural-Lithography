
from cuda_config import *
import numpy as np
import os
import torch
import torch.nn as nn
import cv2
from torch.distributions import Normal
import matplotlib.pyplot as plt
import copy 

def sensor_noise(input, a_poisson=0.004, b_sqrt=0.016):
    """
    Differentiable noise function. Created according to 
    https://pytorch.org/docs/stable/distributions.html
    Noise here = Poisson shot + Gaussian readout 
    """
    # Apply Poisson noise.
    if a_poisson > 0:
        output = torch.poisson(input/a_poisson)*a_poisson
    else:
        output = torch.zeros_like(input)

    # Add Gaussian readout noise.
    read_noise = Normal(loc=torch.zeros_like(output), scale=b_sqrt)
    output = read_noise.sample() + output
    return output

def load_image(file_name: str, normlize_flag=True, torch_sign=True) -> torch.Tensor:
    """Loads the image with OpenCV and converts to torch.Tensor                                      
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)
    
    # load image with OpenCV
    img: np.ndarray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    if normlize_flag:
        img_max = np.max(img[:])
        img_min = np.min(img[:])
        img = (img-img_min)/(img_max-img_min)

    # convert image to torch tensor
    if torch_sign:
        tensor: torch.Tensor = (torch.tensor(img)).to(device)  # CxHxW
        return tensor[None].to(torch.float32)  # 1xCxHxW
    else:
        return img

def normalize(x, mode = 'max'):
    batch_size, num_obj, height, width = x.shape
    x = x.reshape(batch_size, num_obj*height*width)
    if mode == 'max':
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
    elif mode == 'sum':
        x /= x.sum(1, keepdim=True)
    x = x.reshape(batch_size, num_obj, height, width)
    return x

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def center_to_background_ratio(img, centersize=2, shift=0):
    x_center = int(img.shape[-1] / 2)

    half_centersize = int(centersize/2)
    img_back = img.clone()
    img_back[..., (x_center - half_centersize)+shift: (x_center + half_centersize)+shift,
             x_center - half_centersize+shift: x_center + half_centersize+shift] = 0
    mean_of_background = torch.mean(
        img_back, [-3, -2, -1], keepdim=False)  # take mean for each class
    mean_of_center = torch.mean(
        img[..., (x_center - half_centersize)+shift: (x_center + half_centersize)+shift,
            x_center - half_centersize+shift: x_center + half_centersize+shift], [-3, -2, -1], keepdim=False
    )
    pbr = mean_of_center / mean_of_background
    return pbr

def central_crop(variable, tw=None, th=None, dim=2):
    if dim == 2:
        w = variable.shape[-2]
        h = variable.shape[-1]
        if th == None:
            th = tw
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        cropped = variable[..., x1: x1 + tw, y1: y1 + th]
    elif dim == 1:
        h = variable.shape[-1]
        y1 = int(round((h - th) / 2.0))
        cropped = variable[..., y1: y1 + th]
    else:
        raise NotImplementedError
    return cropped

def central_crop3d(variable, ts = None, dim=5):
    if dim >= 3:
        d = variable.shape[-3]
        w = variable.shape[-2]
        h = variable.shape[-1]
        td = ts[-3]
        tw = ts[-2]
        th = ts[-1]
        z1 = int(round((d - td) / 2.0))
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        
        cropped = variable[..., z1: z1 + td, x1: x1 + tw, y1: y1 + th]
    else:
        raise NotImplementedError
    return cropped


def otsu_binarize(slice, visulize=False, erision_flag=False):
    if visulize:
        plt.figure()
        plt.hist(slice.ravel(), 256)
        plt.show()

    grey = copy.deepcopy(slice)
    grey[grey > 160] = 160
    grey = cv2.GaussianBlur(grey, (5, 5), 0)

    kernel_size = 5
    # grey = cv2.bilateralFilter(grey,5,75,75)
    _, bin = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if erision_flag == True:
        sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
        kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        bin = cv2.erode(bin, kernel, iterations=1)
    return bin


class InterpolateComplex2d(nn.Module):
    """Interpolate the complex field in 2D.
    """
    def __init__(self, input_dx, input_field_shape, output_dx, output_field_shape=None, mode='bicubic', del_intermediate_var=False, match_energy_sign=True) -> None:
        super().__init__()
        self.mode = mode
        if output_dx == input_dx:
            pass

        self.input_pad_scale = self.get_input_pad_scale(
            input_dx, input_field_shape, output_dx, output_field_shape)

        self.interpolated_input_field_shape = [
            int(input_dx*side_length*self.input_pad_scale/output_dx) for side_length in input_field_shape[-2:]]

        self.output_field_shape = output_field_shape if output_field_shape is not None else self.interpolated_input_field_shape

        self.del_intermediate_var = del_intermediate_var
        self.match_energy_sign = match_energy_sign

        self.scale_factor = input_dx/output_dx

    def get_input_pad_scale(self, input_dx, input_field_shape, output_dx, output_field_shape):
        if output_field_shape is None:
            input_pad_scale = 1
        else:
            if input_dx * input_field_shape[-2] <= output_dx * output_field_shape[-2]:
                input_pad_scale_x = (
                    output_dx * output_field_shape[-2]) / (input_dx * input_field_shape[-2])
            else:
                input_pad_scale_x = 1

            if input_dx * input_field_shape[-1] <= output_dx * output_field_shape[-1]:
                input_pad_scale_y = (
                    output_dx * output_field_shape[-1]) / (input_dx * input_field_shape[-1])
            else:
                input_pad_scale_y = 1
            input_pad_scale = max(input_pad_scale_y, input_pad_scale_x)

        return input_pad_scale

    def interp_complex(self, x):
        x_in_real_imag = torch.view_as_real(x)  # shape [..., w, h, 2]
        x_real_interpolated = F.interpolate(x_in_real_imag[..., 0], (
            self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]), mode=self.mode, align_corners=False)

        x_imag_interpolated = F.interpolate(x_in_real_imag[..., 1], (
            self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]), mode=self.mode, align_corners=False)

        x_interpolated = torch.stack(
            [x_real_interpolated, x_imag_interpolated], dim=-1)

        if self.del_intermediate_var:
            del x_real_interpolated
        if self.del_intermediate_var:
            del x_imag_interpolated
        x_interpolated = torch.view_as_complex(x_interpolated)
        return x_interpolated

    def circular_pad_or_crop(self, x):
        binary_ouputs = torch.tensor(
            x.shape[-2:]) < torch.tensor(self.output_field_shape)

        intermediate_size = binary_ouputs * \
            (torch.tensor(self.output_field_shape) -
             torch.tensor(x.shape[-2:])) + torch.tensor(x.shape[-2:])
        x = circular_pad(
            x, w_padded=intermediate_size[-2].item(), h_padded=intermediate_size[-1].item())

        x = central_crop(
            x, tw=self.output_field_shape[-2], th=self.output_field_shape[-1])


        return x

    def forward(self, x):
        x = circular_pad(x, pad_scale=self.input_pad_scale)
        x_interpolated = self.interp_complex(x)
        x = x_interpolated/(self.scale_factor)

        if self.del_intermediate_var:
            del x_interpolated
        # central crop to get the desired ouput shape
        if self.del_intermediate_var:
            pass

        if torch.prod(torch.tensor(x.shape[-2:]) >= torch.tensor(self.output_field_shape)):
            output = central_crop(x,
                                  tw=self.output_field_shape[-2], th=self.output_field_shape[-1])
        elif torch.prod(torch.tensor(x.shape[-2:]) < torch.tensor(self.output_field_shape)):
            output = circular_pad(x,
                                  w_padded=self.output_field_shape[-2], h_padded=self.output_field_shape[-1])
        else:
            output = self.circular_pad_or_crop(x)
        if self.del_intermediate_var:
            del x
        return output

def circular_pad(u, pad_scale=None, w_padded=None, h_padded=None,):
    """circular padding last two dimension of a tensor"""
    w, h = u.shape[-2], u.shape[-1]
    if pad_scale != None:
        w_padded, h_padded = w*pad_scale, h*pad_scale
    ww = int(round((w_padded - w) / 2.0))
    hh = int(round((h_padded - h) / 2.0))
    p2d = (hh, hh, ww, ww)
    u_padded = F.pad(u, p2d, mode="constant", value=0)
    return u_padded


def pad_crop_to_size(u, dst_size):
    src_size = list(u.shape[-2:])
    if src_size >= dst_size:
        u_out = central_crop(
        u, dst_size[-2], dst_size[-1])
    elif src_size < dst_size:
        u_out = circular_pad(
                u, w_padded=dst_size[-2],h_padded=dst_size[-1])
    else:
        raise 'wrong input size with src_size{} and dst size {}'.format(src_size, dst_size)
    return u_out


def conv2d(obj, psf, shape="same", intensity_output=False):
    """
    Torch 2D Spatial convolution implemented in Fourier domain.
    - Padding step is necessary; otherwise it will be nonlinear circular convolution.
    - See more in https://ww2.mathworks.cn/matlabcentral/answers/38066-difference-between-conv-ifft-fft-when-doing-convolution
    """

    _, _, im_height, im_width = obj.shape
    output_size_x = obj.shape[-2] + psf.shape[-2] - 1
    output_size_y = obj.shape[-1] + psf.shape[-1] - 1

    p2d_psf = (0, output_size_y -
               psf.shape[-1], 0, output_size_x - psf.shape[-2])
    p2d_obj = (0, output_size_y -
               obj.shape[-1], 0, output_size_x - obj.shape[-2])
    psf_padded = F.pad(psf, p2d_psf, mode="constant", value=0)
    obj_padded = F.pad(obj, p2d_obj, mode="constant", value=0)

    obj_fft = torch.fft.fft2(obj_padded)
    otf_padded = torch.fft.fft2(psf_padded)

    frequency_conv = obj_fft * otf_padded
    convolved = torch.fft.ifft2(frequency_conv)

    if shape == "same":
        convolved = central_crop(convolved, im_height, im_width)
    else:
        raise NotImplementedError

    if intensity_output:
        convolved = torch.abs(convolved)

    return convolved
