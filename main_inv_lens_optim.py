

#%%
""" For the inverse design purpose. 
    Here we use pre-calibrated model to find a better lens design. 
"""
from config import *
from param.param_inv_design_imaging import optim_param, settings, metalens_optics_param
from trainer.mbo_lens import MBOLens

from torchvision.transforms.functional import center_crop
import os
from utils.visualize_utils import show
from utils.general_utils import load_image, normalize
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%%
def load_target(num_imgs, pattern_size, lens_size):
    """ load objects to optimize the lens. Since we only use 4 images to optimize the regression task, we use a function instead of a torch dataloader. 

    Args:
        num_imgs: num of images for the optimization
        obj_size: size of obj
        lens_size: size of lens

    Returns:
        batch of objects.
    """
    image_dir = './data/objs/'
    image_names = sorted(os.listdir(image_dir))
    inputs = []
    for i in range(num_imgs):
        mask_instance = load_image(image_dir + image_names[i])[None]
        mask_instance = center_crop(mask_instance, output_size=[400, 400])[0]        
        inputs.append(mask_instance)
    inputs = torch.stack(inputs)
    inputs = F.interpolate(
        inputs, scale_factor=pattern_size/400, mode='bicubic', align_corners=False)
    if pattern_size != lens_size:
        ind = int((lens_size - pattern_size)//2)
        target = torch.zeros([num_imgs, 1, lens_size, lens_size])
        target[:, :, ind:ind+pattern_size,
               ind:ind+pattern_size] = inputs[:, :1, :, :]
    else:
        target = inputs[:, :1, :, :]
    
    print(target.shape)
    target = normalize(target)
    return target.to(device)


objs = load_target(num_imgs=4, pattern_size=1200, lens_size =1200)
show(objs[0, 0], 'target')
print(objs.max(), objs.min())

#%%

lens_optimizer = MBOLens(
                    optim_param['model_choice'], 
                    settings['use_litho_model_flag'],
                    optim_param['num_iters'], 
                    optim_param['source_mask_optim_lr'],
                    optim_param['use_scheduler'], 
                    optim_param['image_visualize_interval'],
                    metalens_optics_param['cam_a_poisson'],
                    metalens_optics_param['cam_b_sqrt'],
                    save_dir=optim_param['save_dir'],
                    loss_type=metalens_optics_param['loss_type'],
                    )

optimized_doe, print_pred = lens_optimizer.optim(objs)
