

from config import *
import math
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR
from utils.general_utils import cond_mkdir
from utils.visualize_utils import show, plot_loss
from utils.model_utils import model_selector


class FwdOpticsTrainer(object):
    def __init__(self, trainer_param) -> None:
        
        self.add_img_vis = trainer_param['add_img_vis']
        self.image_visualize_interval = trainer_param['image_visualize_interval']
        self.clipping_value = trainer_param['clipping_value']
        self.early_stop_patience = trainer_param['early_stop_patience']
        self.model_update_epochs = trainer_param['model_update_epochs']
        
        # IMPORTANT: which model to use for the forward fitting. 
        self.model_choice = trainer_param['model_choice']
        self.model = model_selector(trainer_param['model_choice'])
        
        self.model_criterion = nn.SmoothL1Loss(beta=trainer_param['loss_beta'], reduction='mean')  # 0.1     
        self.model_update_epochs = trainer_param['model_update_epochs']
        self.use_scheduler = trainer_param['use_scheduler']

        
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=trainer_param['model_lr'])
        self.exp_scheduler = ExponentialLR(self.model_optimizer, gamma=0.99)

        self.save_model_check_point = trainer_param['save_model_check_point']

    def perform_evaluation(self, batch_sample):
        
        batch_images = batch_sample['afm'].to(device) 
        batch_masks = batch_sample['mask'].to(device)
        
        self.model_optimizer.zero_grad()
        images_pred = self.model(batch_masks)
        loss = self.model_criterion(images_pred, batch_images)                

        return loss, batch_images, images_pred
    
    def train_model(self, train_loader):

        self.model.train()
        for param in self.model.parameters():
            param.requries_grad = True
        train_epoch_loss = 0.0

        for _, batch_sample in (enumerate(train_loader)):

            loss, batch_images, images_pred = self.perform_evaluation(batch_sample)
            loss.backward()
            
            if self.clipping_value is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clipping_value
                    )
                    
            self.model_optimizer.step()
            train_epoch_loss += loss
                
        if self.use_scheduler:
            self.exp_scheduler.step()

        train_epoch_loss = train_epoch_loss / (len(train_loader))
        
        return train_epoch_loss, batch_images, images_pred

    def test_model(self, test_loader):
        self.model.eval()
        for param in self.model.parameters():
            param.requries_grad = False
        
        with torch.no_grad():
            test_loss = 0
            for _, batch_sample in enumerate(test_loader):

                batch_loss, batch_images, images_pred_last_batch = self.perform_evaluation(batch_sample)
                test_loss += batch_loss

            test_loss = test_loss / (len(test_loader))
        
        return test_loss, batch_images, images_pred_last_batch

    def fit(self, train_loader, val_loader):
        
        train_losses = []
        val_losses = []
        itr_list = []
        start_time_train = datetime.now()
        best_val_loss = math.inf
        
        for i in range(self.model_update_epochs):

            # Train step
            train_loss, train_last_batch_images, train_last_batch_images_pred = self.train_model(
                train_loader)
            
            # re-evaludate train_loader 
            train_loss, train_last_batch_images, train_last_batch_images_pred = self.test_model(
                train_loader)
            
            # Val step
            val_loss, val_last_batch_images, val_last_batch_images_pred = self.test_model(
                val_loader)

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            itr_list.append(i)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                es = 0
                if self.save_model_check_point:
                    print('save model with val loss {} at epoch {}'.format(val_loss, i))
                    cond_mkdir('model/ckpt/')
                    torch.save(self.model.state_dict(),
                               'model/ckpt/' + "learned_litho_model_"+self.model_choice+".pt")
            else:
                es += 1
                print("Counter {} of 5".format(es))
                if es > self.early_stop_patience:
                    print("Early stop model training with best_val_loss:{} at epoch {}".format(
                        best_val_loss, i))
                    break
                
            if (i+1) % self.image_visualize_interval == 0:
                plot_loss(itr_list, train_losses, filename="train_loss")
                plot_loss(itr_list, val_losses, filename='val_loss')
                
                if self.add_img_vis: 
                    show(train_last_batch_images[0, 0], 'train_gt', cmap='jet')
                    show(train_last_batch_images_pred[0, 0].detach(
                    ).cpu().numpy(), 'train_pred', cmap='jet')
                    
                    show(val_last_batch_images[0, 0], 'val_gt', cmap='jet')
                    show(val_last_batch_images_pred[0, 0].detach(
                    ).cpu().numpy(), 'val_pred', cmap='jet')

        torch.save([train_losses, val_losses],
                   'model/ckpt/' + self.model_choice + "_loss.pt")
        
        end_time_train = datetime.now()
        
        print("duration_train: {}", (end_time_train - start_time_train))
