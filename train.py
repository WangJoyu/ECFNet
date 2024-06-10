# By Yuxiang Sun, Dec. 4, 2019
# Email: sun.yuxiang@outlook.com

import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset, MFNetTrainSet, MFNetDataValSet, FMB_dataset
from util.PST_dataset import PST_dataset
from util.nyud_dataset import nyud_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results, downsample, cosine_similarity_loss
from util.statistics_label import Statistic
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from util.args import ArgumentParserRGBTSegmentation
from RGBT_Net import RGBTNet
#from dual_model import DualModel #as DualModel2
from thop import profile
from ptflops import get_model_complexity_info


augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.2, prob=1.0),
    RandomCropOut(crop_rate=0.1, prob=1.0),
    #RandomBrightness(bright_range=0.15, prob=0.9),
    #RandomNoise(noise_range=5, prob=0.9),
]


def normalization(x):
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1)
    x = x - torch.min(x, dim=-1, keepdim=True)[0]
    #x = x / torch.max(x, dim=-1, keepdim=True)[0]
    #x = 100 * x.reshape(B, C, H, W)
    x = x.reshape(B, C, H, W)
    return x
    
    
# def compute_loss(x, y, num_classes=9):
#     """
#     :param x: [B,Nc,H,W]
#     :param y: [B,H,W]
#     :return: Tensor
#     """
#     losses = F.cross_entropy(x, y, reduction='none')
#     result = []
#     for cls in range(num_classes):
#         mask = y == cls
#         l = losses[mask]
#         if l.shape[0]:
#             result.append(l.mean())
#     #print(result)
#     loss = torch.stack(result).mean() + losses.mean()
#     return loss


def train(args, epo, model, train_loader, optimizer, teacher=None):
    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = images.cuda(args.gpu)
        labels = labels.long().cuda(args.gpu)
        start_t = time.time()  # time.time() returns the current time
        optimizer.zero_grad()
        logits = model(images)
        
        #print(images.shape, labels.shape, logits[0].shape)
        if isinstance(logits, list):
            loss1 = F.cross_entropy(logits[0], labels, ignore_index=-1) #+ 10*cosine_similarity_loss(F.softmax(logits[0], dim=1), pse_labels)  #, weight=torch.tensor([0.22, 0.85, 1.0, 1.09, 1.09, 1.09, 1.4, 1.11, 1.11]).to('cuda:1'))
            loss2 = F.cross_entropy(logits[1], labels, ignore_index=-1) #+ 10*cosine_similarity_loss(F.softmax(logits[1], dim=1), pse_labels)
            loss3 = F.cross_entropy(logits[2], labels, ignore_index=-1) #+ 10*cosine_similarity_loss(F.softmax(logits[2], dim=1), pse_labels)
            loss4 = F.cross_entropy(logits[3], labels, ignore_index=-1) #+ 10*cosine_similarity_loss(F.softmax(logits[3], dim=1), pse_labels)
            loss = loss1 + loss2 + loss3 + loss4
        else:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)

        loss.backward()
        optimizer.step()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']

        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, loss %.4f, time %s' \
            % (args.model_name, epo, args.epochs, it+1, len(train_loader), lr_this_epo, float(loss),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True  # note that I have not colorized the GT and predictions here
        accIter['train'] = accIter['train'] + 1


def validation(args, epo, model, val_loader):
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time()  # time.time() returns the current time
            logits = model(images)
            loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function

            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (args.model_name, epo, args.epochs, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits[0].argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1


def testing(args, epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            #ther = Variable(ther).cuda(args.gpu)
            #logits = model(images.type(torch.cuda.FloatTensor))
            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=range(0,args.n_class)) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.model_name, epo, args.epochs, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU = compute_results(conf_total)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, backbone: %s+%s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.encoder_1, args.encoder_2, args.lr, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write('%3d: |' % epo)
        for i in range(len(precision)):
            f.write('%7.4f, %7.4f,| ' % (100*recall[i], 100*IoU[i]))
        f.write('%7.4f, %7.4f\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)
    return 100*np.mean(np.nan_to_num(IoU))


if __name__ == '__main__':
    parser = ArgumentParserRGBTSegmentation(
        description='Efficient RGBT outdoor Sematic Segmentation (Training with pytorch)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    print("the model name:", args.model_name)
    
    #model1 = eval(args.model_name)(num_classes=args.n_class, encoder_1=args.encoder_1, encoder_2=args.encoder_2)
    #model1.load_state_dict(torch.load(f'./runs/{args.model_name}/2022-09-19-17/68.pth'))
    #input_ = torch.randn(1,4,480,640)
    #flops, params = profile(model1, inputs=(input_,))
    #flops, params = get_model_complexity_info(model1, (4, 480, 640), as_strings=True, print_per_layer_stat=False)
    #print('Flops:  ' + flops)
    #print('Params: ' + params)
    #print('params: %.2f M| flops: %2.f G' % (params/10**6, flops))#/10**9))
    #del model1
    #del input_
    model = eval(args.model_name)(num_classes=args.n_class, encoder_1=args.encoder_1, encoder_2=args.encoder_2)
    if args.gpu >= 0:
        model.cuda(args.gpu)

    #para = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=args.lr, step_size_up=5, step_size_down=35, mode='triangular2', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

    weight_dir = os.path.join("./runs", args.model_name, datetime.date.today().isoformat()+'-'+str(datetime.datetime.now().hour))
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
 
    writer = SummaryWriter("./runs/tensorboard_log")

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    # print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(data_dir=args.dataset_dir, split='train', transform=augmentation_methods)
    # val_dataset = MF_dataset(data_dir=args.dataset_dir, split='val')
    test_dataset = MF_dataset(data_dir=args.dataset_dir, split='test')
    
    # train_dataset = FMB_dataset(data_dir=args.dataset_dir, split='train', transform=augmentation_methods)
    # test_dataset = FMB_dataset(data_dir=args.dataset_dir, split='test')

    # train_dataset = PST_dataset(data_dir=args.dataset_dir, split='train', is_mirror=True, is_scale=True)
    # test_dataset = PST_dataset(data_dir=args.dataset_dir, split='test')
    # Training and validation sets
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    '''
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )'''
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    if args.epochs_from != 0:
        weight_dir = os.path.join('./runs/', args.model_name, args.time_path)
        # model.load_state_dict(torch.load(os.path.join(weight_dir, str(args.epochs_from - 1) + '.pth')))
        loading_path = os.path.join(weight_dir, str(args.epochs_from - 1) + '.pth')
        model.load_state_dict(torch.load(loading_path))
        print('loading weights: {}'.format(loading_path))
    best_miou =0
    for epo in range(args.epochs_from, args.epochs):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
                    
        train(args, epo, model, train_loader, optimizer)
        #validation(args, epo, model, val_loader)

        cur_miou = testing(args, epo, model, test_loader) # testing is just for your reference, you can comment this line during training
        if cur_miou > best_miou:
            checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
            print('saving check point %s: ' % checkpoint_model_file)
            torch.save(model.state_dict(), checkpoint_model_file)
            best_miou = cur_miou
        scheduler.step()