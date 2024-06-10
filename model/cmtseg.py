import torch
import math
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from resnet import *
import segformer
from model.class_mask_transformer import Transformer


class CMTSeg(nn.Module):
    def __init__(self, backbone, num_classes=19, img_size=(480, 640), hidden_dim=256,
                 aux_loss=False, pretrained_on_imagenet=True):
        """
        Initializes the model.
        :param backbone: resnet, see backbone.py
        :param transformer: the transformer architecture. See transformer.py
        :param num_classes: number of object classes
        :param num_queries: resoultion of the feature
        :param aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        #self.backbone = eval(backbone)(pretrained_on_imagenet=True)
        #self.backbone_ir = eval(backbone)(pretrained_on_imagenet=True)
        self.backbone = getattr(segformer, backbone)()
        self.backbone_ir = getattr(segformer, backbone)()
        if pretrained_on_imagenet:
            state_dict = torch.load(backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            #state_dict = expand_state_dict(self.encoder_2.state_dict(), state_dict)
            self.backbone.load_state_dict(state_dict, strict=True)
            self.backbone_ir.load_state_dict(state_dict, strict=True)
            print('loading {} weights'.format(backbone))
        self.class_embed = nn.Conv2d(hidden_dim*4, num_classes, 1)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        features_size = [[img_size[0] // (4 * 2 ** i), img_size[1] // (4 * 2 ** i)] for i in range(4)]
        
        # self.num_queries = features_size[0][0] * features_size[0][1]
        # self.query_embed = nn.Parameter(torch.randn(self.num_queries, hidden_dim))
        # features_size = [[120, 160], [60, 80], [30, 40], [15, 20]]
        self.cls_tokens = nn.Parameter(torch.randn(num_classes, hidden_dim))
        #embed_dims = [self.backbone.down_4_channels_out, self.backbone.down_8_channels_out,
        #              self.backbone.down_16_channels_out, self.backbone.down_32_channels_out]
        embed_dims = self.backbone.embed_dims
        self.transformer = Transformer(embed_dims=embed_dims, hidden_dim=hidden_dim, num_class=num_classes)
        # embed_dims = [64, 128, 256, 512]
        #self.pos_embed = [PositionEmbeddingLearned(*feature_size, feature_dim) for feature_size, feature_dim in
        #                  zip(features_size, embed_dims)]
        self.aux_loss = aux_loss
        self.loss = CMTSegLoss(w_focal=20, w_dice=1)
        self.img_size = img_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, images, labels=None):
        rgb, ir = images[:, 0:3], images[:, 3:]
        ir = torch.cat([ir, ir, ir], dim=1)
        features_rgb = self.backbone(rgb)
        features_ir = self.backbone_ir(ir)
        features = [rgb_ + ir_ for rgb_, ir_ in zip(features_rgb, features_ir)]
        # features = self.backbone(images)
        #pos_embed = [pos_fn(feature) for pos_fn, feature in zip(self.pos_embed, features)]
        output, masks = self.transformer(features, self.cls_tokens, pos_embed=None)
        masks = [F.interpolate(mask, size=self.img_size, mode='bilinear', align_corners=True) for mask in masks]
        masks = torch.stack(masks, 0).sum(dim=0)
        output = self.class_embed(output)
        output = F.interpolate(output, size=self.img_size, mode='bilinear', align_corners=True)
        if self.training:
            assert labels is not None, 'please input labels'
            return self.loss({'pred_logits': output, 'pred_masks': masks}, labels)
        else:
            return masks


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, h, w, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(h, num_pos_feats // 2)
        self.col_embed = nn.Embedding(w, num_pos_feats // 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        self.col_embed.to(x.device)
        self.row_embed.to(x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class CMTSegLoss(nn.Module):
    def __init__(self, w_focal: float = 1., w_dice: float = 1., w_class: float = 1.):
        super(CMTSegLoss, self).__init__()
        self.w_class = w_class
        self.w_focal = w_focal
        self.w_dice = w_dice

    def class_loss(self, pred_logits, target_logits):
        class_loss = F.cross_entropy(pred_logits, target_logits)
        return class_loss

    def focal_loss(self, predict, target, gamma=2.0, alpha=None):
        # predict : b, c, h, w
        # target : b, h, w
        logpt = -F.cross_entropy(predict, target, reduction='none')  # b, h, w

        p_t = torch.exp(logpt)  # b, h, w
        focal_loss = -((1 - p_t) ** gamma) * logpt
        if alpha:
            alpha = alpha[None, :, None, None].expand(predict.shape)
            alpha_t = alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

    def dice_loss(self, predict, target):
        smooth = 1e-8

        input_flat = predict.flatten(1)
        target_flat = target.flatten(1)

        intersection = input_flat * target_flat

        #loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 2 * intersection.sum(1) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.mean()

        return loss

    def multi_class_dice_loss(self, predict, target, weights=None):
        B, C, H, W = predict.shape
        predict = predict.softmax(1)
        #predict = predict.sigmoid()
        target = F.one_hot(target, C).permute(0, 3, 1, 2)
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        total_loss = 0

        for i in range(1, C):
            dice_loss = self.dice_loss(predict[:, i], target[:, i])
            if weights is not None:
                dice_loss *= weights[i]
            total_loss += dice_loss

        return total_loss / (C-1)

    def forward(self, out, targets):
        pred_logits = out["pred_logits"]  # b, n, class + 1
        pred_masks = out["pred_masks"]  # b, n, h, w

        class_loss = self.class_loss(pred_logits, targets) * self.w_class
        focal_loss = self.focal_loss(pred_masks, targets) * self.w_focal
        #dice_loss = self.class_loss(pred_masks, targets) * self.w_dice
        dice_loss = self.multi_class_dice_loss(pred_masks, targets) * self.w_dice

        return class_loss, focal_loss, dice_loss
