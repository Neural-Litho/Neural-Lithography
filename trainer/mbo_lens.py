

""" model based optimization for    
"""

from config import *
import cv2
import numpy as np
from torch.optim.lr_scheduler import StepLR
from kornia.losses import SSIMLoss, PSNRLoss

from optics.free_space_fwd import FreeSpaceFwd 
from optics.doe import DOE
from param.param_inv_design_imaging import metalens_optics_param
from param.param_fwd_litho import litho_param
from model.learned_litho import model_selector
from utils.visualize_utils import show, plot_loss
from utils.general_utils import normalize, center_to_background_ratio, central_crop, sensor_noise, conv2d
from utils.img_processing import torch_richardson_lucy_fft


class MBOLens(object):
    def __init__(self, model_choice, use_litho_model_flag, num_iters, lr, use_scheduler, image_visualize_interval, cam_a_poisson, cam_b_sqrt, save_dir='', loss_type=None) -> None:
        
        self.model_choice = model_choice
        self.use_litho_model_flag = use_litho_model_flag
        self.cam_a_poisson = cam_a_poisson
        self.cam_b_sqrt = cam_b_sqrt
        self.litho_model = model_selector(model_choice)
        
        # the psf of lens shares the same path with the holography task.
        self.lens_model = FreeSpaceFwd(
            metalens_optics_param['input_dx'], metalens_optics_param['input_shape'],
            metalens_optics_param['output_dx'], metalens_optics_param['output_shape'],
            metalens_optics_param['lambda'], metalens_optics_param['z'], 
            metalens_optics_param['pad_scale'], metalens_optics_param['Delta_n']
            )
        
        self.doe = DOE(
                       metalens_optics_param['num_partition'],
                       metalens_optics_param['num_level'], 
                       metalens_optics_param['input_shape'], 
                       litho_param['slicing_distance'],
                       doe_type=metalens_optics_param['doe_type']
                       )
        
        self.load_pretrianed_model(use_litho_model_flag)
        
        self.loss_type = loss_type
        self.num_iters = num_iters
        self.lr = lr
        self.mask_optimizer = torch.optim.AdamW(
            [self.doe.logits], lr=self.lr)
        
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)  # 0.1
        self.image_visualize_interval = image_visualize_interval
        self.save_dir = save_dir
        self.metric_ssim = SSIMLoss(window_size=1)
        self.metric_psnr = PSNRLoss(max_val=1)

        for param in self.litho_model.parameters():
            param.requries_grad = False

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

    def load_pretrianed_model(self, use_litho_model_flag):
        
        print('load_pretrained_model_for_optimize is {}'.format(
            use_litho_model_flag))
        
        if use_litho_model_flag:
            checkpoint = torch.load(
                'model/ckpt/' + "learned_litho_model_"+ self.model_choice + ".pt")
            self.litho_model.load_state_dict(checkpoint)
            for param in self.litho_model.parameters():
                param.requries_grad = False

    def forward_imaging(self, batch_target, itr):
        
        # get psf
        mask = self.doe.get_doe_sample()
        if self.use_litho_model_flag:
            print_pred = self.litho_model(mask)
        else:
            print_pred = mask
            
        psf = torch.abs(self.lens_model(print_pred))**2
        psf_sum = torch.sum(psf)
        
        if torch.isnan(psf).any():
            raise
        
        # get sensor(camera) image
        sensor_img = conv2d(batch_target, psf, intensity_output=True)
        sensor_img = sensor_img + sensor_noise(sensor_img, 0.004, 0.02)

        # get loss
        loss, deconv_img, metric_ssim, metric_psnr = self.calculate_loss(sensor_img, batch_target, psf)

        return loss, metric_ssim, metric_psnr, mask, psf, psf_sum, deconv_img, sensor_img, print_pred
    
    
    def optim(self, batch_target):
        loss_list = []
        itr_list = []
        for i in range(self.num_iters):
            self.mask_optimizer.zero_grad()

            loss, mssim, mpsnr, mask, psf, psf_intensity_sum, deconv_img, sensor_img, print_pred = self.forward_imaging(batch_target, i)
            
            loss.backward()
            self.mask_optimizer.step()
            if self.use_scheduler:
                self.scheduler.step()
                
            loss_list.append(loss.item())
            itr_list.append(i)

            psf_save = self.visualize(i, mask, sensor_img, psf, deconv_img,
                           itr_list, loss_list, mssim, mpsnr, psf_intensity_sum, loss)

        # save optimized psf and mask_to_fab
        mask_logits = self.doe.logits_to_doe_profile()[0]
        mask_to_save = (mask_logits.detach().cpu().numpy()+10).astype(np.uint8)
        psf_to_save = psf_save.numpy()
        cv2.imwrite(self.save_dir+'/mask'+'.bmp', mask_to_save)
        cv2.imwrite(self.save_dir+'/psf'+'.bmp',
                    (psf_to_save*255).astype(np.uint8))

        return mask_logits, print_pred
