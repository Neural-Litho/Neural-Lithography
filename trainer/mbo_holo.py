

""" Model based optimization for the holographic optical element (HOE).
"""
import torch
import torch.nn as nn
import cv2
import numpy as np

from task.free_space_fwd import FreeSpaceFwd
from task.doe import DOE
from param.param_inv_design_holography import holo_optics_param, litho_param
from litho.learned_litho import model_selector
from utils.visualize_utils import show, plot_loss
from utils.general_utils import cond_mkdir, normalize

from torch.optim.lr_scheduler import ReduceLROnPlateau
from kornia.losses import SSIMLoss


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


class MBOHolo(object):
    """ Model based optimization for the 
    The models are 'litho model' + 'task (holo) model'.
    """
    def __init__(self, model_choice, use_litho_model_flag, num_iters, lr, 
                 use_scheduler, image_visualize_interval, save_dir='', eff_weight=0.1) -> None:

        self.num_iters = num_iters
        self.holo_pipeline = HoloPipeline(model_choice, use_litho_model_flag)

        self.mask_optimizer = torch.optim.Adam(
            [self.holo_pipeline.doe.logits], lr=lr)

        self.loss_fn = nn.MSELoss()
        self.eff_weight = eff_weight
        self.image_visualize_interval = image_visualize_interval
        self.save_dir = save_dir
        cond_mkdir(self.save_dir)

        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.mask_optimizer, 'min')

    def hoe_loss(self, holo_intensity, target):
        
        N_img = torch.sum(target)  # number of pixels in target
        
        I_avg = torch.sum(holo_intensity*target)/N_img  # avg of img region
        
        rmse_loss = torch.sqrt(self.loss_fn(holo_intensity/I_avg, target))
        eff = torch.sum(holo_intensity*target) / \
            (torch.prod(torch.tensor(target.shape[-2:])))
        
        loss = rmse_loss + (1-eff) * self.eff_weight
        return loss

    def optim(self, batch_target):

        loss_list = []
        itr_list = []

        for i in range(self.num_iters):
            
            self.mask_optimizer.zero_grad()
            holo_intensity, holo_sum, mask = self.holo_pipeline()

            loss = self.hoe_loss(
                holo_intensity, batch_target)
            loss.backward()
            self.mask_optimizer.step()
            if self.use_scheduler:
                self.scheduler.step(loss)
            
            loss_list.append(loss.item())
            itr_list.append(i)

            if (i + 1) % self.image_visualize_interval == 0:
                show(mask[0, 0].detach().cpu(),
                     'doe mask at itr {}'.format(i), cmap='jet')
                target = normalize(holo_intensity)[0, 0].detach().cpu()

                show(target, 'intensity at itr {} is {}'.format(
                    i, holo_sum), cmap='gray')
                plot_loss(itr_list, loss_list, filename="loss")

        mask_logits = self.holo_pipeline.doe.logits_to_doe_profile()[0]
        mask_to_save = (mask_logits.detach().cpu().numpy()+10).astype(np.uint8)
        cv2.imwrite(self.save_dir+'/mask'+'.bmp', mask_to_save)

        ssim_fun = SSIMLoss(window_size=1)
        metric_ssim = 1-ssim_fun(normalize(holo_intensity), batch_target)*2
        print('SSIM between target and image is:', metric_ssim)
        
        return mask
