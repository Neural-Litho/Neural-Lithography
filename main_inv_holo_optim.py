

# %%
""" Inverse design the HOE. 
    Here we use pre-trained neural litho digital twin to find a better doe layout to send to fab.
"""
from cuda_config import device
from trainer.mbo_holo import MBOHolo
from param.param_inv_design_holography import optim_param, settings
from utils.visualize_utils import show
from utils.general_utils import load_image


holo_target = load_image('data/target.bmp', normlize_flag=True)[None].to(device)

show(holo_target[0, 0], 'target')

# %%
# initialize the optimizer
hoe_optimizer = MBOHolo(optim_param['model_choice'], 
                        settings['use_litho_model_flag'],
                        optim_param['num_iters'], 
                        optim_param['source_mask_optim_lr'],
                        optim_param['use_scheduler'], 
                        optim_param['image_visualize_interval'],
                        save_dir=optim_param['save_dir'],
                        )

# optimize the hoe
optimized_hoe = hoe_optimizer.optim(holo_target)


