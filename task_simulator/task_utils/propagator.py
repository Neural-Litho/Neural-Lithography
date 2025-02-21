
"""optical propagator under rayleigh-sommerfeld approximation

"""

import torch
import torch.nn as nn
import math
from utils.general_utils import circular_pad, InterpolateComplex2d


class RSCProp(nn.Module):
    """     
    Rayleigh-Sommerfeld convolution
    """

    def __init__(self,
                 input_dx,
                 input_field_shape,
                 output_dx=None,
                 output_field_shape=None,
                 wave_lengths=None,
                 z=0,  # prop dist, key param here
                 pad_scale=2.,
                 pre_compute_H=True,  # set False when H is learnable
                 ) -> None:

        super().__init__()
        self.pad_scale = pad_scale
        self.pre_compute_H = pre_compute_H
        self.output_dx = output_dx

        output_field_shape = output_field_shape if output_field_shape is not None else input_field_shape
        self.output_field_shape = output_field_shape

        self.H = self.get_prop_kernel(
            z, input_field_shape, input_dx, wave_lengths)

        self.interpolate_complex_2d = InterpolateComplex2d(
            input_dx, [input_field_shape[i]*pad_scale for i in range(
                len(input_field_shape))], output_dx, output_field_shape)

    def get_prop_kernel(self, distance, field_shape, dx, wavelength):
        M, N = field_shape[-2], field_shape[-1],
        x = torch.linspace(-M*dx/(2), M*dx/2, int(M))
        y = torch.linspace(-N*dx/(2), N*dx/2, int(N))

        meshx, meshy = torch.meshgrid(x, y, indexing='ij')

        r = torch.sqrt(meshx**2+meshy**2+distance**2)
        h = (1/r-1j*2*math.pi/wavelength)*torch.exp(1j*2 *
                                                    math.pi*r/wavelength)*distance/2/math.pi/r**2

        # pad kernel to avoid error from circular convolution
        if self.pad_scale is not None:
            h = circular_pad(h, self.pad_scale)

        H = torch.fft.fftshift(torch.fft.fft2(
            torch.fft.fftshift(h, dim=[-2, -1])), dim=[-2, -1]) * dx**2
        return H

    def forward(self, field, match_shape=True):

        u1 = circular_pad(field, self.pad_scale)
        U1 = torch.fft.fftshift(torch.fft.fft2(
            torch.fft.fftshift(u1, dim=[-2, -1])), dim=[-2, -1])

        U2 = U1 * self.H.to(U1.device)
        u2 = torch.fft.ifftshift(torch.fft.ifft2(
            torch.fft.ifftshift(U2, dim=[-2, -1])), dim=[-2, -1])

        # interpolate in case input_dx is not equal to output_dx
        if match_shape:
            u2 = self.interpolate_complex_2d(u2)
        return u2
