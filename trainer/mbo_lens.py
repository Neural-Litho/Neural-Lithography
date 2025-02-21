

"""  model-based optimization for imaging lens design.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.optim.lr_scheduler import StepLR
from kornia.losses import SSIMLoss, PSNRLoss
from param.param_inv_design_imaging import metalens_optics_param
from param.param_fwd_litho import litho_param
from litho_simulator.learned_litho import model_selector
from utils.visualize_utils import show, plot_loss
from utils.general_utils import normalize, center_to_background_ratio, central_crop
from task_simulator.task_utils.reconstruction import torch_richardson_lucy_fft




class MBOLens(object):
    """ Co-design through two diff simulators:
        ---pretrained-litho ---- imaging lens (to be optimized) ---
    """
    def __init__(self, model_choice, use_litho_model_flag, num_iters, lr, use_scheduler, image_visualize_interval, cam_a_poisson, cam_b_sqrt, save_dir='', loss_type=None) -> None:
        
        self.model_choice = model_choice
        
        self.cam_a_poisson = cam_a_poisson
        self.cam_b_sqrt = cam_b_sqrt
        
        # init the litho model
        self.litho_model = model_selector(model_choice)
        # load  pretrained params for the litho model
        self.load_pretrained_litho_model(use_litho_model_flag)

        # init the camera model
        self.camera = CameraPipeline(metalens_optics_param, litho_param, use_litho_model_flag)
        
        # init the optimization process
        self.initialize_optimization(lr, num_iters, loss_type, use_scheduler, image_visualize_interval, save_dir)

    def load_pretrained_litho_model(self, use_litho_model_flag):
        print('load_pretrained_model_for_optimize is {}'.format(
            use_litho_model_flag))
        
        if use_litho_model_flag:
            checkpoint = torch.load(
                'model/ckpt/' + "learned_litho_model_"+ self.model_choice + ".pt")
            self.litho_model.load_state_dict(checkpoint)
            for param in self.litho_model.parameters():
                param.requries_grad = False

    def initialize_optimization(self, lr, num_iters, loss_type, use_scheduler, image_visualize_interval, save_dir):
        self.loss_type = loss_type
        self.num_iters = num_iters
        self.lr = lr
        self.mask_optimizer = torch.optim.AdamW(
            [self.camera.doe.logits], lr=self.lr)
        
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)  # 0.1
        self.image_visualize_interval = image_visualize_interval
        self.save_dir = save_dir
        self.metric_ssim = SSIMLoss(window_size=1)
        self.metric_psnr = PSNRLoss(max_val=1)

        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = StepLR(
                self.mask_optimizer, step_size=25, gamma=0.5)
    
    def visualize(self, i, mask, sensor_img, psf, deconv_img, itr_list, loss_list, mssim, mpsnr, psf_sum, loss):
        psf_save = None
        if (i + 1) % self.image_visualize_interval == 0:
            show(mask[0, 0].detach().cpu(),
                    'doe mask at itr {}'.format(i), cmap='jet')
            psf_save = central_crop(
                normalize(psf)[0, 0].detach().cpu(), 128)
            show(psf_save, 'psf at itr {} is {}'.format(i, psf_sum), cmap='gray')
            show((sensor_img)[0, 0].detach().cpu(),
                    'sensor_img at itr {}'.format(i), cmap='gray')
            if deconv_img is not None:
                show((deconv_img)[0, 0].detach().cpu(),
                        'deconv_img at itr {}'.format(i), cmap='gray')
            plot_loss(itr_list, loss_list, filename="loss")
            print('loss is {} at itr {}'.format(loss, i))
            print('SSIM and PSNR is {} and {} at itr {}.'.format(mssim, mpsnr, i))
        return psf_save
    
    def calculate_loss(self, cam_img, target, psf):
        deconv_result = None
        metric_ssim1 = 1-self.metric_ssim(cam_img, target)*2
        metric_psnr1 = -self.metric_psnr(cam_img, target)
        metric_ssim = [metric_ssim1.item()]
        metric_psnr = [metric_psnr1.item()]
        
        if self.loss_type == 'cbr':
            # direct imaging
            loss = -torch.log(center_to_background_ratio(psf, centersize=10))
            
        elif self.loss_type == 'deconv_loss':
            # computational imaging, which uses RL deconvolution; here we embed the deconv process into the loss calculation
            deconv_result = torch_richardson_lucy_fft(cam_img, psf)                
            loss = self.loss_fn(deconv_result, target)
            metric_ssim2 = 1-self.metric_ssim(deconv_result, target)*2
            metric_psnr2 = -self.metric_psnr(deconv_result, target)
            metric_ssim.append(metric_ssim2.item())
            metric_psnr.append(metric_psnr2.item())         
        else:
            print('wrong type {}'.format(self.loss_type))
            raise Exception

        return loss, deconv_result, metric_ssim, metric_psnr
    
    def save_optimized_psf_mask(self, psf_save):
        # save optimized psf and mask_to_fab
        mask_logits = self.camera.doe.logits_to_doe_profile()[0]
        mask_to_save = (mask_logits.detach().cpu().numpy()+10).astype(np.uint8)
        psf_to_save = psf_save.numpy()
        cv2.imwrite(self.save_dir+'/mask'+'.bmp', mask_to_save)
        cv2.imwrite(self.save_dir+'/psf'+'.bmp',
                    (psf_to_save*255).astype(np.uint8))
        return mask_logits
    
    def optim(self, batch_target):
        loss_list = []
        itr_list = []
        for i in range(self.num_iters):
            self.mask_optimizer.zero_grad()
            sensor_img, psf, psf_sum, print_pred, mask = self.camera(batch_target, self.litho_model)
            loss, deconv_img, metric_ssim, metric_psnr = self.calculate_loss(sensor_img, batch_target, psf)
            
            loss.backward()
            self.mask_optimizer.step()
            if self.use_scheduler:
                self.scheduler.step()
                
            loss_list.append(loss.item())
            itr_list.append(i)

            psf_save = self.visualize(i, mask, sensor_img, psf, deconv_img,
                           itr_list, loss_list, metric_ssim, metric_psnr, psf_sum, loss)

        mask_logits = self.save_optimized_psf_mask(psf_save)
        return mask_logits, print_pred
