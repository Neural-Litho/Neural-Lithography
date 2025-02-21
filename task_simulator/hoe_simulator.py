
"""
    This file is used to simulate the holography, i.e., the design of holographic optical element (HOE).
"""



import torch
import torch.nn as nn

from task_simulator.task_utils.free_space_fwd import FreeSpaceFwd
from task_simulator.task_utils.doe import DOE
from param.param_inv_design_holography import holo_optics_param, litho_param
from litho_simulator.learned_litho import model_selector


class HoloPipeline(nn.Module):
    
    """ Co-design through two diff simulators:
        ---litho ---- Holo ---
    """
    def __init__(self, model_choice, use_litho_model_flag) -> None:
        super().__init__()
        
        self.model_choice = model_choice
        self.litho_model = model_selector(model_choice)
        self.use_litho_model_flag = use_litho_model_flag
        
        if use_litho_model_flag:
            print('load_pretrained_model_for_optimize is {}'.format(
            use_litho_model_flag))
            self.load_pretrianed_model()

        # init a parameterized DOE
        self.doe = DOE(holo_optics_param['num_partition'], 
                       holo_optics_param['num_level'],
                       holo_optics_param['input_shape'], 
                       litho_param['slicing_distance'],
                       doe_type='2d')
        
        # init a holography system
        self.optical_model = FreeSpaceFwd(
            holo_optics_param['input_dx'], holo_optics_param['input_shape'], 
            holo_optics_param['output_dx'], holo_optics_param['output_shape'],
            holo_optics_param['lambda'], holo_optics_param['z'], 
            holo_optics_param['pad_scale'], holo_optics_param['Delta_n'])

    def load_pretrianed_model(self):
        checkpoint = torch.load(
            'model/ckpt/' + "learned_litho_model_"+ self.model_choice + ".pt")
        self.litho_model.load_state_dict(checkpoint)
        for param in self.litho_model.parameters():
            param.requries_grad = False

    def forward(self):
        
        mask = self.doe.get_doe_sample()

        if self.use_litho_model_flag:
            print_pred = self.litho_model(mask)
        else:
            print_pred = mask

        holo_output = self.optical_model(print_pred)
        holo_intensity = torch.abs(holo_output)**2
        holo_sum = torch.sum(holo_intensity)

        return holo_intensity, holo_sum, mask
