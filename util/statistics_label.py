import torch
import torch.nn.functional as F


class Statistic():
    def __init__(self, args, beta=0.999):
        self.beta = beta
        self.num_classes = args.n_class
        self.gpu = args.gpu
        self.probability = torch.eye(self.num_classes).cuda(args.gpu)

    def get_probability(self):
        return self.probability

    def update_probability(self, predict, label):
        B, _, H, W = predict.shape
        predict = F.softmax(predict, dim=1)
        pre_label = torch.max(predict, dim=1)[1]
        eye = torch.eye(self.num_classes).cuda(self.gpu)
        one_hot_label = eye[label].permute(0, 3, 1, 2)  # B num_class H W
        one_hot_pre = eye[pre_label].permute(0, 3, 1, 2)
        for ind in range(self.num_classes):
            pre_class = one_hot_pre[:, ind]
            label_class = one_hot_label[:, ind]
            true_pre = (pre_class + label_class) == 2
            # true_pre = true_pre.unsqueeze(1).expand(1, self.num_classes, H, W)
            probability = predict.permute(0, 2, 3, 1)[true_pre, :]
            if probability.shape[0]:
                self.probability[ind] = self.beta * self.probability[ind] + (1-self.beta) * probability.mean(dim=0)
            else:
                continue

    def get_pseudo_label(self, predict, label, lamda=0.9):
        B, _, H, W = predict.shape
        predict = F.softmax(predict, dim=1)
        pre_label = torch.max(predict, dim=1)[1]  # B H W
        pseudo_label = predict.permute(0, 2, 3, 1)  # B H W num_class
        mask = label != pre_label  # B H W
        pseudo_label[mask, :] = self.probability[label[mask]]
        pseudo_label[~mask, :] = self.probability[label[~mask]]
        pseudo_label = pseudo_label.permute(0, 3, 1, 2)
        pseudo_label = lamda*pseudo_label + (1-lamda)*predict

        return pseudo_label


if __name__ == '__main__':
    S = Statistic(5)
    label = torch.randint(0, 5, (2, 8, 8))
    pre = torch.rand((2, 5, 8, 8))
    pre_gt = F.softmax(pre, dim=1)
    S.update_probability(pre_gt, label)
    p = S.get_probability()
    pse = S.get_pseudo_label(pre_gt, label)
    print(p, pse)
