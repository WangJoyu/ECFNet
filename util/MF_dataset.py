# By Yuxiang Sun, Jul. 3, 2021
# Email: sun.yuxiang@outlook.com

import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
import glob
import cv2
import random

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted', 'train_task2'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image
    
    def normalize(self, img):  #, mean, std):
        # pytorch pretrained model need the input range: 0-1
        mean = np.array([0.485, 0.456, 0.406, 0.449], dtype=np.float32).reshape(4,1,1)  # np.array([0.22156, 0.25873, 0.23003, 0.39541])
        std = np.array([0.229, 0.224, 0.225, 0.226], dtype=np.float32).reshape(4,1,1)   # np.array([0.16734, 0.16907, 0.16801, 0.07578])
        #img = img.astype(np.float32) / 255.0
        #print(type(img), type(mean))
        img = img - mean
        img = img / std
        return img
        
    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')
        
        for func in self.transform:
            image, label = func(image, label)
            
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)))
        image = image.astype('float32')
        image = np.transpose(image, (2, 0, 1))/255.0
        image = self.normalize(image)
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')
        '''
        if self.split == 'train':
            label_4 = self.read_image(name, 'labels_4').astype('int64')
            label_8 = self.read_image(name, 'labels_8').astype('int64')
            label_16 = self.read_image(name, 'labels_16').astype('int64')
            label_32 = self.read_image(name, 'labels_32').astype('int64')
            binary = self.read_image(name, 'binary')
            binary = np.asarray(PIL.Image.fromarray(binary).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
            binary = binary.astype('int64')
            return torch.tensor(image), torch.tensor(label), torch.tensor(label_4), torch.tensor(label_8), torch.tensor(label_16), torch.tensor(label_32), name'''

        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data


class FMB_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[]):
        super(FMB_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted', 'train_task2'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image
    
    def normalize(self, img):  #, mean, std):
        # pytorch pretrained model need the input range: 0-1
        mean = np.array([0.485, 0.456, 0.406, 0.449], dtype=np.float32).reshape(4,1,1)  # np.array([0.22156, 0.25873, 0.23003, 0.39541])
        std = np.array([0.229, 0.224, 0.225, 0.226], dtype=np.float32).reshape(4,1,1)   # np.array([0.16734, 0.16907, 0.16801, 0.07578])
        #img = img.astype(np.float32) / 255.0
        #print(type(img), type(mean))
        img = img - mean
        img = img / std
        return img
        
    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        thermal = self.read_image(name, 'thermal')
        label = self.read_image(name, 'labels')
        #print(image.shape, thermal.shape)
        image = np.concatenate((image, thermal[:,:,:1]), axis=-1)
        for func in self.transform:
            image, label = func(image, label)
            
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)))
        image = image.astype('float32')
        image = np.transpose(image, (2, 0, 1))/255.0
        image = self.normalize(image)
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')
        '''
        if self.split == 'train':
            label_4 = self.read_image(name, 'labels_4').astype('int64')
            label_8 = self.read_image(name, 'labels_8').astype('int64')
            label_16 = self.read_image(name, 'labels_16').astype('int64')
            label_32 = self.read_image(name, 'labels_32').astype('int64')
            binary = self.read_image(name, 'binary')
            binary = np.asarray(PIL.Image.fromarray(binary).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
            binary = binary.astype('int64')
            return torch.tensor(image), torch.tensor(label), torch.tensor(label_4), torch.tensor(label_8), torch.tensor(label_16), torch.tensor(label_32), name'''

        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data
        
        
class MFNetTrainSet(Dataset):
    def __init__(self, root, max_iters=None, crop_size=(480, 640), scale=True, mirror=True, mask=True, ignore_label=-1):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.is_scale = scale
        self.is_mirror = mirror
        self.is_mask = mask
        self.ignore_label = ignore_label

        img_folder = os.path.join(root, 'images')
        mask_folder = os.path.join(root, 'labels')
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.files = []
        for basename in self.names:
            filename = basename + '.png'
            imgpath = os.path.join(img_folder, filename)
            maskpath = os.path.join(mask_folder, filename)
            self.files.append({
                "img": imgpath,
                "label": maskpath,
                "name": filename
            })
        #self._scribbles = sorted(glob.glob("../dataset/SCRIBBLES/*.png"))[::-1][:1000]
        self._scribbles = sorted(glob.glob("../dataset/SCRIBBLES/*.png"))[:1000]  #(0.01,0.1] (0.1,0.2] (0.2,0.3] (0.3,0.4] (0.4,0.5] (0.5,0.6] 各2000，共12000

        if max_iters:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))
            self.files = self.files[:max_iters]

        print('{} training images are loaded!'.format(len(self.files)))

        self.num_class = 9
        self.label2name = ['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']

    def __len__(self):
        return len(self.files)
    
    def normalize(self, img):  #, mean, std):
        # pytorch pretrained model need the input range: 0-1
        mean = np.array([0.485, 0.456, 0.406, 0.449], dtype=np.float32).reshape(4,1,1)  # np.array([0.22156, 0.25873, 0.23003, 0.39541])
        std = np.array([0.229, 0.224, 0.225, 0.226], dtype=np.float32).reshape(4,1,1)   # np.array([0.16734, 0.16907, 0.16801, 0.07578])
        #img = img.astype(np.float32) / 255.0
        #print(type(img), type(mean))
        img = img - mean
        img = img / std
        return img

    def generate_scale_label(self, image, label):
        f_scale = 1. + random.randint(0, 10) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        scribble_path = self._scribbles[random.randint(0, 999)]
        image = np.asarray(PIL.Image.open(datafiles["img"]))
        label = np.asarray(PIL.Image.open(datafiles["label"]))
        scribble = np.asarray(PIL.Image.open(scribble_path).convert("P"))
        scribble = cv2.resize(scribble, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST)
        # image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        # label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        size = image.shape
        #print(size, scribble.shape)

        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        # image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        image = image/255
        image = image.transpose((2, 0, 1))
        image = self.normalize(image)
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[:, h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        #image = image.transpose((2, 0, 1))
        #print(image.shape, label.shape)
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        if self.is_mask and random.randint(0, 1):
            image = image * (np.max(scribble) - scribble)
        # label = label - 1
        label = label.astype('int64')
        return image.copy(), label.copy(), name


class MFNetDataValSet(Dataset):
    def __init__(self, root, ignore_label=-1):
        self.root = root
        self.ignore_label = ignore_label
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        self.files = []

        img_folder = os.path.join(root, 'images')
        mask_folder = os.path.join(root, 'labels')
        with open(os.path.join(root, 'test.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.files = []
        for basename in self.names:
            filename = basename + '.png'
            imgpath = os.path.join(img_folder, filename)
            maskpath = os.path.join(mask_folder, filename)
            self.files.append({
                "img": imgpath,
                "label": maskpath,
                "name": filename
            })

        print('{} validation images are loaded!'.format(len(self.files)))

        self.num_class = 9

    def __len__(self):
        return len(self.files)
    
    def normalize(self, img):  #, mean, std):
        # pytorch pretrained model need the input range: 0-1
        mean = np.array([0.485, 0.456, 0.406, 0.449], dtype=np.float32).reshape(4,1,1)  # np.array([0.22156, 0.25873, 0.23003, 0.39541])
        std = np.array([0.229, 0.224, 0.225, 0.226], dtype=np.float32).reshape(4,1,1)   # np.array([0.16734, 0.16907, 0.16801, 0.07578])
        #img = img.astype(np.float32) / 255.0
        #print(type(img), type(mean))
        img = img - mean
        img = img / std
        return img

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = np.asarray(PIL.Image.open(datafiles["img"]))
        label = np.asarray(PIL.Image.open(datafiles["label"]))
        # image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        # label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        size = image.shape

        name = datafiles['name']
        image = np.asarray(image, np.float32)
        # image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        image = image/255
        image = image.transpose((2, 0, 1))
        image = self.normalize(image)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # label = label - 1
        label = label.astype('int64')

        return image.copy(), label.copy(), name