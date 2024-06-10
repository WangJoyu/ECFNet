import os

import numpy as np
import cv2
import torch
from PIL import Image
from cnnformer import WeTry
from RGBT_Net import RGBTNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from util.MF_dataset import MF_dataset, FMB_dataset
import tqdm
import PIL
#from main import visualize


palette = np.array([(0, 0, 0), (173, 229, 229), (187, 57, 134), (45, 163, 178), (206, 176, 47), 
(131, 54, 200), (56, 171, 83), (183, 71, 78), (66, 102, 167), (14, 127, 255), 
(138, 163, 91), (156, 98, 153), (101, 153, 140), (225, 214, 155), (136, 111, 89)])

def visualize(pred):
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
        img[pred == cid] = palette[cid]
    img = np.uint8(img)
    return img

if __name__ == '__main__':
    torch.cuda.set_device(0)
    model = RGBTNet(num_classes=15, pretrained_on_imagenet=False).to('cuda:0')
    state_dict = torch.load('./runs/RGBTNet/2024-02-04-18/32.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    #test_dataset = MF_dataset(data_dir='../RTFNet-master/dataset/', split='val_test')
    test_dataset = FMB_dataset(data_dir='../dataset/FMB_Dataset/', split='test')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    save_path = './result'
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(tqdm.tqdm(test_loader)):
            images = Variable(images).cuda(0)
            logits = model(images)
            prediction = logits.argmax(1).cpu().numpy().squeeze()
            labels = labels.cpu().numpy().squeeze()
            #prediction = np.asarray(PIL.Image.fromarray(prediction).resize((800, 600), resample=PIL.Image.NEAREST))
            pre_img = visualize(prediction)
            pre_img = cv2.resize(pre_img, (800, 600), interpolation=cv2.INTER_NEAREST)
            
            predict = Image.fromarray(pre_img)
            predict.save(os.path.join(save_path, 'FMB', names[0]))
            #  cv2.imwrite(os.path.join(save_path, 'prediction', names[0]+'.png'), pre_img)
            '''
            labels = visualize(labels)
            predict = Image.fromarray(labels)
            predict.save(os.path.join(save_path, 'labels', names[0]+'.png'))'''
            # cv2.imwrite(os.path.join(save_path, 'labels', names[0]+'.png'), labels)

