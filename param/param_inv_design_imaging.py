


settings = {
    "use_litho_model_flag": True,
}

# optim param
optim_param = {
    'model_choice': 'fno', # from {pbl3d, fno, physics}
    'use_scheduler': True,
    'num_iters': 2000,
    'source_mask_optim_lr':  1e0,  
    'save_dir': 'data/holo_exp/hologram_to_print',
    'image_visualize_interval':50,
}

# parameter for metalens inv design
metalens_optics_param = {
    'input_dx': 0.1, # 0.1 um/pixel
    'input_shape': [1200, 1200], # 1600 for single fov, 3000 for large aperture
    'output_dx': 0.1, # 0.1 um/pixel
    'output_shape': [1200, 1200], # 1600 for single fov, 3000 for large aperture
    'lambda': 0.633, # red light
    'z': 400,  # NA 0.15
    'pad_scale': 2,
    'num_level': 12, 
    'num_partition': 1200,
    'loss_type': 'deconv_loss',  # deconv_loss, cbr
    'doe_type':'1d', # 2d or 1d; 1d means the doe is rotational symmetrical
    'cam_a_poisson':.004, 
    'cam_b_sqrt':0.02,
    'Delta_n': 0.545,  # refractive index
}
