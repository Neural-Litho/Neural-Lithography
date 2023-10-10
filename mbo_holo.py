

""" model based optimization for the inverse design. 
use the:
    - pre-defined opc model
    - pre-learned printing degradation model 
    to inverse optimize the printing mask
"""
from config import *
import torch
import torch.nn as nn
import cv2
import numpy as np

from optics import HoloFwd, DOE
from param.param_inv_design_holography import holo_optics_param
from utils.model_utils import model_selector
from utils.visualize_utils import show, plot_loss
from utils.general_utils import cond_mkdir, normalize

from torch.optim.lr_scheduler import ReduceLROnPlateau
from kornia.losses import SSIMLoss


class HoloPipeline(nn.Module):
    def __init__(self, model_choice, use_litho_model_flag) -> None:
        super().__init__()

        self.litho_model = model_selector(model_choice)
        self.use_litho_model_flag = use_litho_model_flag
        self.load_pretrianed_model(use_litho_model_flag)

        self.doe = DOE(holo_optics_param['num_partition'], 
                       holo_optics_param['num_level'],
                       holo_optics_param['input_shape'], 
                       doe_type='2d')

        self.optic_model = HoloFwd(
            holo_optics_param['input_dx'], holo_optics_param['input_shape'], 
            holo_optics_param['output_dx'], holo_optics_param['output_shape'],
            holo_optics_param['lambda'], holo_optics_param['z'], 
            holo_optics_param['pad_scale'], holo_optics_param['Delta_n'])

    def load_pretrianed_model(self, use_litho_model_flag):
        
        print('load_pretrained_model_for_optimize is {}'.format(
            use_litho_model_flag))
        
        if use_litho_model_flag:
            checkpoint = torch.load(
                'model/ckpt/' + "learned_litho_model_pbl3d.pt")
            self.litho_model.load_state_dict(checkpoint)
            for param in self.litho_model.parameters():
                param.requries_grad = False

    def forward(self):
        
        mask = self.doe.get_doe_sample()

        if self.use_litho_model_flag:
            print_pred = self.litho_model(mask*100)/100
        else:
            print_pred = mask

        holo_output = self.optic_model(print_pred)
        holo_intensity = torch.abs(holo_output)**2
        holo_sum = torch.sum(holo_intensity)

        return holo_intensity, holo_sum, mask


class MBOHolo(object):
    def __init__(self, model_choice, use_litho_model_flag, num_iters, lr, 
                 use_scheduler, image_visualize_interval=50, save_dir='') -> None:

        self.num_iters = num_iters
        self.lr = lr
        self.holo_pipeline = HoloPipeline(model_choice, use_litho_model_flag)

        self.mask_optimizer = torch.optim.Adam(
            [self.holo_pipeline.doe.logits], lr=self.lr)

        self.loss_fn = nn.MSELoss()

        self.image_visualize_interval = image_visualize_interval
        self.save_dir = save_dir
        cond_mkdir(self.save_dir)

        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.mask_optimizer, 'min')

    def calculate_loss(self, holo_intensity, target, target_binarized):
        
        N_img = torch.sum(target_binarized)  # number of pixels in target
        I_avg = torch.sum(holo_intensity*target_binarized)/N_img # avg of img region
        
        rmse_loss = torch.sqrt(self.loss_fn(holo_intensity/I_avg, target))
        eff = torch.sum(holo_intensity*target_binarized)/(1024**2)
        
        loss = rmse_loss + (1-eff) *0.1

        return loss

    def optim(self, batch_target=None, target_binarized=None):

        loss_list = []
        itr_list = []

        for i in range(self.num_iters):
            
            self.mask_optimizer.zero_grad()

            holo_intensity, holo_sum, mask = self.holo_pipeline()

            loss = self.calculate_loss(
                holo_intensity, batch_target, target_binarized)

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

        metric1 = SSIMLoss(window_size=1)
        metric_ssim = 1-metric1(normalize(holo_intensity), target_binarized)*2
        print('SSIM between target and image is:', metric_ssim)
        
        return mask
