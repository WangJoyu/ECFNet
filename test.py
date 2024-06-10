import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
from util.MF_dataset import MF_dataset
import tqdm
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from RGBT_Net import RGBTNet

if __name__ == '__main__':
    torch.cuda.set_device(0)
    model = RGBTNet(num_classes=9, pretrained_on_imagenet=False).to('cuda:0')
    state_dict = torch.load('./runs/RGBTNet/RGB-T 2022-10-08-9/78.pth', map_location='cpu')
    # state_dict = torch.load('./233.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    # print(model)
    # image = torch.randn(1, 3, 480, 640)
    # thermal = torch.randn(1, 1, 480, 640)

    test_dataset = MF_dataset(data_dir='../RTFNet-master/dataset/', split='test')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    model.eval()
    conf_total = np.zeros((9, 9))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(tqdm.tqdm(test_loader)):
            images = images.cuda(0)
            labels = labels.long().cuda(0)
            #images = Variable(images).cuda(0)
            #labels = Variable(labels).cuda(0)
            logits = model(images)

            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
    precision, recall, IoU = compute_results(conf_total)
    print("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %)")

    for i in range(len(precision)):
        print('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]), end='')
    print('%0.4f, %0.4f' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))

