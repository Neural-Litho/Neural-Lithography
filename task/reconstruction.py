

import torch


def torch_ift(input, dim=2):
    """ifft with fftshift
    NOTE:  implementation of torch fft is tricky
    """
    if dim == 2:
        output = torch.fft.ifftshift(torch.fft.ifft2(
            torch.fft.ifftshift(input, [-2, -1])), [-2, -1])
    elif dim == 3:
        print('input', input.shape)
        output = torch.fft.ifftshift(torch.fft.ifftn(
            torch.fft.ifftshift(input, [-3, -2, -1]), dim=[0, 1, 2]), [-3, -2, -1])
        print('output', output.shape)
    else:
        raise NotImplementedError
    return output


def torch_ft(input, dim=2, size=None):
    """fft with fftshift
    NOTE:  implementation of torch fft is tricky
    """
    if dim == 2:
        if size is not None:
            output = torch.fft.fftshift(torch.fft.fft2(
                torch.fft.fftshift(input, [-2, -1]), size), [-2, -1])
        else:
            output = torch.fft.fftshift(torch.fft.fft2(
                torch.fft.fftshift(input, [-2, -1])), [-2, -1])
    elif dim == 3:
        print('input', input.shape)
        output = torch.fft.fftshift(torch.fft.fftn(
            torch.fft.fftshift(input, [-3, -2, -1]), dim=[0, 1, 2]), [-3, -2, -1])
        print('output', output.shape)
    else:
        raise NotImplementedError
    return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torch_richardson_lucy_fft(image, psf, num_iter=50):
    """
    image: 4-dimensional input, NCHW format
    psf:   4-dimensional input, NCHW format
    https://stackoverflow.com/questions/9854312/how-does-richardson-lucy-algorithm-work-code-example
    """

    img_deconv = torch.full(image.shape, 0.5).to(device)
    eps = 1e-12

    psf_fft = torch_ft(psf)
    bp_fft = torch.conj(psf_fft)  # fft of back projector

    for i in range(num_iter):
        img_deconv_fft = torch_ft(img_deconv)
        relative_blur = image/torch_ift(img_deconv_fft*psf_fft)

        img_deconv = img_deconv * torch_ift(torch_ft(relative_blur)*bp_fft)

    return torch.abs(img_deconv)

