

"""store the param for the inverse design 
"""


settings = {
    "use_litho_model_flag": True,
}

optim_param = {
    'model_choice': 'pbl3d', 
    'use_scheduler': True,
    'num_iters': 2000,
    'source_mask_optim_lr':  1e0,  
    'save_dir': 'data/holo_exp/hologram_to_print',
    'image_visualize_interval':50,
}

# parameter for hologram inv design
holo_optics_param = {
    'input_dx': 0.1,
    'input_shape': [1024, 1024],
    'output_dx': 0.1,
    'output_shape': [1024, 1024],
    'lambda': 0.633,
    'z': 300, 
    'pad_scale': 2,
    'num_level': 12,
    'num_partition': 256,
    'Delta_n': 0.545, # refractive index
}

