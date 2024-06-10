import torch
import math
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from resnet import *
from  model.transformer import Transformer


class SegTR(nn.Module):
    def __init__(self, backbone, num_classes=19, img_size=(480, 640), hidden_dim=256, aux_loss=False):
        """
        Initializes the model.
        :param backbone: resnet, see backbone.py
        :param transformer: the transformer architecture. See transformer.py
        :param num_classes: number of object classes
        :param num_queries: resoultion of the feature
        :param aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        #self.num_queries = num_queries
        #self.transformer = Transformer(hidden_dim=hidden_dim)

        self.backbone_rgb = eval(backbone)(pretrained_on_imagenet=True)
        self.backbone_ir = eval(backbone)(pretrained_on_imagenet=True)
        self.class_embeding = nn.Conv2d(hidden_dim, num_classes, 1)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        #self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        #features_size = [[120, 160], [60, 80], [30, 40], [15, 20]]
        features_size = [[img_size[0]//(4 * 2**i), img_size[1]//(4 * 2**i)] for i in range(4)]
        self.transformer = Transformer(hidden_dim=hidden_dim, query_size=features_size[0])
        self.num_queries = features_size[0][0] * features_size[0][1]
        self.query_embeding = nn.Parameter(torch.randn(self.num_queries, hidden_dim))
        
        embed_dims=[64, 128, 256, 512]
        self.pos_embed = [PositionEmbeddingLearned(*feature_size, feature_dim) for feature_size, feature_dim in zip(features_size, embed_dims)]
        self.aux_loss = aux_loss
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

    def forward(self, images):
        rgb, ir = images[:, 0:3], images[:, 3:]
        ir = torch.cat([ir, ir, ir], dim=1)
        features_rgb = self.backbone_rgb(rgb)
        features_ir = self.backbone_ir(ir)
        features = [rgb_ + ir_ for rgb_, ir_ in zip(features_rgb, features_ir)]
        pos_embed = [pos_fn(feature) for pos_fn, feature in zip(self.pos_embed, features)]
        output = self.transformer(features, self.query_embeding, pos_embed)
        
        output = self.class_embeding(output)
        return F.interpolate(output, scale_factor=4)
        
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, h, w, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(h, num_pos_feats//2)
        self.col_embed = nn.Embedding(w, num_pos_feats//2)
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

