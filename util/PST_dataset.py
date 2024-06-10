# By Joyu Wang, Dem. 13, 2022
# Email: Wongjoyu@163.com

import os, torch, random, cv2
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL


class PST_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[], is_mirror=False, is_scale=False):
        super(PST_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted', 'train_task2'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split, split + '.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)
        self.is_mirror = is_mirror
        self.is_scale = is_scale

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s/%s.png' % (self.split, folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def normalize(self, img):  # , mean, std):
        # pytorch pretrained model need the input range: 0-1
        mean = np.array([0.3445, 0.3479, 0.3309, 0.2319]).reshape(4, 1, 1)
        # mean = np.array([0.22156, 0.25873, 0.23003, 0.39541]).reshape(4, 1, 1)
        std = np.array([0.2493, 0.2547, 0.2544, 0.2682]).reshape(4, 1, 1)
        # std = np.array([0.16734, 0.16907, 0.16801, 0.07578]).reshape(4, 1, 1)
        # img = img.astype(np.float32) / 255.0
        img = img - mean
        img = img / std
        # ther = ther - ther_mean
        # ther = ther / ther_std
        # mean = np.array([0.485, 0.456, 0.406, 0.449], dtype=np.float32).reshape(4, 1, 1)
        return img

    def generate_scale_label(self, image, label):
        f_scale = 0.7 + random.randint(0, 8) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'rgb')
        ther = self.read_image(name, 'thermal')
        label = self.read_image(name, 'labels')

        image = np.concatenate([image, np.expand_dims(ther, axis=-1)], axis=2)
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)

        '''if self.split == 'train':
            img_h, img_w = label.shape
            h_off = random.randint(0, img_h - self.input_h)
            w_off = random.randint(0, img_w - self.input_w)
            image = np.asarray(image[h_off: h_off + self.input_h, w_off: w_off + self.input_w], np.float32)
            label = np.asarray(label[h_off: h_off + self.input_h, w_off: w_off + self.input_w], np.int64)
        else:
            image = cv2.resize(image, dsize=[self.input_h, self.input_w], interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=[self.input_h, self.input_w], interpolation=cv2.INTER_NEAREST)'''
        if self.split == 'train':
            img_h, img_w = label.shape
            h_off = random.randint(0, img_h - self.input_h)
            w_off = random.randint(0, img_w - self.input_w)
            image = np.asarray(image[h_off: h_off + self.input_h, w_off: w_off + self.input_w], np.float32)
            label = np.asarray(label[h_off: h_off + self.input_h, w_off: w_off + self.input_w], np.int64)
        else:
            image = cv2.resize(image, dsize=[self.input_h, self.input_w], interpolation=cv2.INTER_LINEAR).astype(np.float32)
            label = cv2.resize(label, dsize=[self.input_h, self.input_w], interpolation=cv2.INTER_NEAREST).astype(np.int64)
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = self.normalize(image)
        #label = np.transpose(label, (2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return torch.tensor(image.copy()), torch.tensor(label.copy()), name

    def __len__(self):
        return self.n_data

