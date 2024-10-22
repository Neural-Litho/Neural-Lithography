

""" Fitting a forward neural litho model corresponding to a real-world photolithography process. 
"""

# %%
from param.param_fwd_litho import trainer_param, dataset_param, litho_param
from data.afm_dataio import afm_dataloader
from trainer.fwd_learned_litho_trainer import FwdLithoTrainer

# load data
train_loader, val_loader = afm_dataloader(
    dataset_param['data_path'],
    litho_param['slicing_distance'],
    dataset_param['batch_size'],
    dataset_param['shuffle'],
    dataset_param['train_sample_ratio'],
    dataset_param['num_data_to_load'],
    dataset_param['random_crop'],
    dataset_param['output_size'],
)

# initialize the trainer
trainer = FwdLithoTrainer(trainer_param)

# train the neural litho model
trainer.fit(train_loader, val_loader)
