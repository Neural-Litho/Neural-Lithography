

"""params for using homography to calibrate the 2 slm and the camera.
"""


homo_settings = {
    'slm_type': 'gaea',  # from  {'pluto', 'gaea'}
    # 'data_dir': 'hardware/homo_calib/homo_data/0404/',
    'normalized_homography_flag': True,
    'data_dir': '../homo_data/',
}

homo_optim_param = {
    'learning_rate': 1e-3,  # was 1e-3,  # the gradient optimisation update step
    'num_iterations': 150,  # the number of iterations until convergence for the l1 loss
    'num_levels': 5,  # the total number of image pyramid levels
    'error_tol': 1e-3,
}
