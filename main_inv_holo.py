

# %%
""" For the inverse design purpose. 
    Here we use pre-calibrated model to find a better doe design. 
"""
from config import *
from mbo_holo import MBOHolo
from param.param_inv_litho import optim_param, settings
from utils.visualize_utils import show
from utils.general_utils import load_image


inputs = load_image('data/target.bmp', normlize_flag=True)[None]
target = inputs.to(device)
target_binarized = target.clone()

show(target[0, 0], 'target')

# %%

doe_optimizer = MBOHolo(optim_param['model_choice'], 
                        settings['use_litho_model_flag'],
                        optim_param['num_iters'], 
                        optim_param['source_mask_optim_lr'],
                        optim_param['use_scheduler'], 
                        optim_param['image_visualize_interval'],
                        save_dir=optim_param['save_dir'],
                        )
optimized_doe = doe_optimizer.optim(target, target_binarized)
