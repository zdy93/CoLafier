import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_pretrained import SinCosLabelEncoding
from torch import Tensor


class LinearModel(nn.Module):
    def __init__(self, num_classes, with_label):
        super().__init__()
        self.num_classes = num_classes
        self.with_label = with_label
        self.input_layer = nn.Linear(784, 256)
        self.fc = nn.Linear(256, num_classes)
        self.label_embedding = nn.Embedding(num_classes, 256)
        self.sin_cos_label_embedding = SinCosLabelEncoding(256)
        self.merge_fc = nn.Linear(2 * 256, 256)
        self.layerNorm = nn.LayerNorm(256)

    def forward(self, x: Tensor, label=None, get_feat=True, merge_concat=False, activation=None, label_2=None,
                lam=None, embedding_type="regular"):
        x = torch.flatten(x, 1)
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        if self.with_label:
            if embedding_type == "regular":
                embedding_layer = self.label_embedding
            else:
                embedding_layer = self.sin_cos_label_embedding
            label_embeds = embedding_layer(label)
            if label_2 is not None:
                label_2_embeds = embedding_layer(label_2)
                label_embeds = lam * label_embeds + (1 - lam) * label_2_embeds
            if merge_concat:
                feature = torch.cat([x, label_embeds], dim=1)
                feature = self.merge_fc(feature)
            else:
                feature = x + label_embeds
            if activation is not None:
                if activation == 'gelu':
                    feature = F.gelu(feature)
                elif activation == 'relu':
                    feature = F.relu(feature)
            feature = self.layerNorm(feature)
            logits = self.fc(feature)
            if get_feat:
                return logits, feature
            else:
                return logits
        else:
            x = self.fc(x)
            return x
