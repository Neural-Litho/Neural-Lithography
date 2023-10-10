

""" Fitting the forward modelling from the lithography results. 
"""

# %%
from config import *
from param.param_fwd_litho import trainer_param, dataset_param
from data.afm_dataio import AFM_dataloader
from model.fwd_learned_trainer import FwdOpticsTrainer

train_loader, val_loader = AFM_dataloader(
    dataset_param['data_path'],
    dataset_param['batch_size'],
    dataset_param['shuffle'],
    dataset_param['train_sample_ratio'],
    dataset_param['num_data_to_load'],
    dataset_param['random_crop'],
    dataset_param['output_size'],
)
trainer = FwdOpticsTrainer(trainer_param)
trainer.fit(train_loader, val_loader)
