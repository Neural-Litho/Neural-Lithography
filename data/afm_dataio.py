
import torch
import os
import json
import itertools
from torch.utils.data import random_split, DataLoader, Dataset
from utils.general_utils import load_image, pad_crop_to_size


class AFMdataset(Dataset):

    def __init__(self, data_path, slicing_distance, num_data_to_load=None,
                 random_crop=False, output_size=(256, 256),
                 load_folder='', map_location=torch.device('cpu')):

        self.output_size = output_size
        self.slicing_distance = slicing_distance
        self.mask_dir = data_path + load_folder+'mask/'
        self.afm_dir = data_path + 'afm/'
        self.afm_ug_dir = data_path + 'afm_ug/'  # un-registered afm data

        self.random_crop = random_crop
        self.mask_names = sorted(os.listdir(self.mask_dir))
        self.afm_names = self.mask_names
        self.max_height_dict = json.load(open(data_path+"src_max_dict.txt"))

        if num_data_to_load is not None:
            self.mask_names = self.mask_names[:num_data_to_load]
            self.afm_names = self.afm_names[:num_data_to_load]
            self.max_height_dict = dict(itertools.islice(
                self.max_height_dict.items(), num_data_to_load))

        self.map_location = map_location

    def tpl_levels_to_heights(self, mask, afm, idx):
        # converts the mask levels to the real height
        mask *= self.slicing_distance  # 1 level corresponds to 0.1um
        
        # height in the dict are in m, we convert it to um
        afm = afm / 255 * self.max_height_dict[self.mask_names[idx]] * 1e6
        
        return mask, afm

    def __len__(self):
        length = len(self.mask_names)
        return length

    def __getitem__(self, idx):

        mask_level = load_image(os.path.join(
            self.mask_dir, self.mask_names[idx]), normlize_flag=False).to(torch.float32)
        afm_level = load_image(os.path.join(
            self.afm_dir, self.afm_names[idx]), normlize_flag=False).to(torch.float32)

        mask_height, afm_height = self.tpl_levels_to_heights(mask_level, afm_level, idx)
        mask_height = pad_crop_to_size(mask_height, [256, 256])
        afm_height = pad_crop_to_size(afm_height, [256, 256])

        sample = {
            'mask': mask_height,
            'afm': afm_height,
        }

        return sample


def afm_dataloader(data_path, batch_size, shuffle, sample_ratio_to_train_and_val,
                   num_data_to_load, random_crop, output_size):

    afm_set = AFMdataset(data_path,
                         num_data_to_load,
                         random_crop,
                         output_size,)

    train_set_size = int(len(afm_set) * sample_ratio_to_train_and_val)
    valid_set_size = len(afm_set) - train_set_size

    train_dataset, val_dataset = random_split(
        afm_set, [train_set_size, valid_set_size])

    print('='*30)
    print('Train data set:', len(train_dataset))
    print('Test data set:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader
