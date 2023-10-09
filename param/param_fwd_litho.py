

dataset_param = {
    'data_path': 'data/our_printed_data/230403_nanoscribe_GDS_AFM_12/',
    'batch_size': 4,
    'shuffle': False,
    'train_sample_ratio': 0.75,
    'num_data_to_load': None,
    'random_crop': True,
    'output_size':(256,256),
}

trainer_param = {
    'model_choice': 'fno',  # choose from 'pbl3d', 'fno' and 'physics'
    'save_model_check_point': True,
    'model_update_epochs': 4000,
    'model_lr': 5e-5, #1e-4 for fno, 5e-4 for pbl3d, 1e-2 for physics
    'loss_beta': 0.1, #useful for the Huber loss
    'add_img_vis': True, 
    'image_visualize_interval': 100,
    'clipping_value': None,
    'early_stop_patience': 20,
    'use_scheduler': False,
    'use_ensemble': False,
    'num_ensembles':1, # should >1
}
