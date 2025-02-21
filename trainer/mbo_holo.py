

""" Model based optimization for the holographic optical element (HOE).
"""
import torch
import torch.nn as nn
import cv2
import numpy as np

from utils.visualize_utils import show, plot_loss
from utils.general_utils import cond_mkdir, normalize

from torch.optim.lr_scheduler import ReduceLROnPlateau
from kornia.losses import SSIMLoss
from task_simulator.hoe_simulator import HoloPipeline

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
