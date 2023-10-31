

""" Forward fitting the neural lithography simulator based on the collected dataset.
"""

dataset_param = {
    'data_path': 'data/printed_data/',
    'batch_size': 4,
    'shuffle': False,
    'train_sample_ratio': 0.75, # the rest is for validation
    'num_data_to_load': None,
    'random_crop': True,
    'output_size':(256,256),
}

litho_param = {
    'hatching_distance': 0.1, # um 
    'slicing_distance': 0.1, # um 
}

trainer_param = {
    'model_choice': 'fno',  # choose from 'pbl3d', 'fno' and 'physics'
    'save_model_check_point': True,
    'model_update_epochs': 4000,
    'model_lr': 1e-4, # 1e-4 for fno, 5e-4 for pbl3d, 1e-3 for physics
    'loss_beta': 0.1, # useful when using the Huber loss
    'add_img_vis': True, 
    'image_visualize_interval': 100,
    'clipping_value': None,
    'early_stop_patience': 20,
    'use_scheduler': False,
    # 'use_ensemble': False,
    # 'num_ensembles':1, # should >1
}
