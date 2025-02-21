

import torch
import torch.nn as nn

from task_simulator.task_utils.free_space_fwd import FreeSpaceFwd
from task_simulator.task_utils.doe import DOE
from utils.general_utils import sensor_noise, conv2d


class CameraPipeline(nn.Module):
    """ Forward camera model for the image formation of meta/diffractive lens imaging with designed layout.
    """
    def __init__(self, metalens_optics_param, litho_param, use_litho_model_flag) -> None:
        super(CameraPipeline, self).__init__()
        
        self.use_litho_model_flag = use_litho_model_flag
        
        # init the doe profile 
        self.doe = DOE(
                metalens_optics_param['num_partition'],
                metalens_optics_param['num_level'], 
                metalens_optics_param['input_shape'], 
                litho_param['slicing_distance'],
                doe_type=metalens_optics_param['doe_type']
                )
        
        # the psf of lens in the imaging task shares the same propagation path with the holography task.
        self.lens_model = FreeSpaceFwd(
            metalens_optics_param['input_dx'], metalens_optics_param['input_shape'],
            metalens_optics_param['output_dx'], metalens_optics_param['output_shape'],
            metalens_optics_param['lambda'], metalens_optics_param['z'], 
            metalens_optics_param['pad_scale'], metalens_optics_param['Delta_n']
            )
    
    def get_psf(self, litho_model):
        # get psf
        mask = self.doe.get_doe_sample()
        if self.use_litho_model_flag:
            print_pred = litho_model(mask)
        else:
            print_pred = mask
        
        psf = torch.abs(self.lens_model(print_pred))**2
        psf_sum = torch.sum(psf)
        
        if torch.isnan(psf).any():
            raise
        
        return psf, psf_sum, print_pred, mask
    
    def forward(self, batch_target, litho_model):
        psf, psf_sum, print_pred, mask = self.get_psf(litho_model)
        
        # get sensor(camera) image
        sensor_img = conv2d(batch_target, psf, intensity_output=True)
        sensor_img = sensor_img + sensor_noise(sensor_img, 0.004, 0.02)

        return sensor_img, psf, psf_sum, print_pred, mask 
        