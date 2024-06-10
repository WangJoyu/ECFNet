# By Joyu Wang, Dem. 20, 2023
# Email: Wongjoyu@163.com

import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL


class nyud_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[]):
        super(nyud_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted', 'train_task2'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, name)
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def normalize(self, img):  # , mean, std):
        # pytorch pretrained model need the input range: 0-1
        mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).reshape(6, 1, 1)
        #mean = np.array([0.485, 0.456, 0.406, 0.449]).reshape(4, 1, 1)
        # mean = np.array([0.22156, 0.25873, 0.23003, 0.39541]).reshape(4, 1, 1)
        std = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).reshape(6, 1, 1)
        #std = np.array([0.229, 0.224, 0.225, 0.226]).reshape(4, 1, 1)
        # std = np.array([0.16734, 0.16907, 0.16801, 0.07578]).reshape(4, 1, 1)
        img = img.astype(np.float32) / 255.0
        img = img - mean
        img = img / std
        # mean = np.array([0.485, 0.456, 0.406, 0.449], dtype=np.float32).reshape(4, 1, 1)
        return img

    def __getitem__(self, index):
        name = self.names[index].split()
        image = self.read_image(name[0], 'rgb')
        ther = self.read_image(name[0].replace('RGB', 'HHA'), 'depth')
        label = self.read_image(name[1], 'labels')
        
        #print(image.shape, ther.shape)
        image = np.concatenate([image, ther], axis=2)

        for func in self.transform:
            image, label = func(image, label)

        img , ther = image[:, :, :3], image[:, :, 3:]
        img = np.asarray(PIL.Image.fromarray(img).resize((self.input_w, self.input_h)))
        ther = np.asarray(PIL.Image.fromarray(ther).resize((self.input_w, self.input_h)))
        image = np.concatenate([img, ther], axis=2)
        image = image.astype('float32')
        image = np.transpose(image, (2, 0, 1))
        image = self.normalize(image)
        #img, ther = image[:3], image[3:] 

        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')-1
        # if self.split == 'train':
        #     binary = self.read_image(name, 'binary')
        #     binary = np.asarray(PIL.Image.fromarray(binary).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        #     binary = binary.astype('int64')
        #     return torch.tensor(image), torch.tensor(label), torch.tensor(binary), name
        #print(image.shape, label.shape)

        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data