import torch
import torch.nn as nn
from glow.vae_model import *

class Feature_Extractor(object):
    def __init__(self, data_device, hparams):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(data_device) if torch.cuda.is_available() else 'cpu')
        feature_extractor = torch.load(hparams.Train.feature_extractor, map_location=self.device).to(self.device)
        encoder = feature_extractor.encoder.to(self.device)
        self.encoder = encoder
        
        for param in encoder.parameters():
            param.requires_grad = False

        network = list(encoder.children())[0]
        all_layers = []
        for idx, layer in enumerate(network):
            for i, l in enumerate(layer.children()):
                all_layers.append(l)

        self.extractor1, self.extractor2, self.extractor3 = [], [], []
        for idx, layer in enumerate(all_layers):
            if idx <= 3:
                self.extractor1.append(layer)
            if idx > 3 and idx <= 9:
                self.extractor2.append(layer)
            if idx > 9 and idx <= 15:
                self.extractor3.append(layer)
        self.extractor1, self.extractor2, self.extractor3 = nn.Sequential(*self.extractor1), nn.Sequential(*self.extractor2), nn.Sequential(*self.extractor3)


    def get_feature(self, x):
        x = x.reshape(list(x.shape[:2])+[-1]).permute([0, 2, 1])
        f1 = self.extractor1(x)
        f2 = self.extractor2(f1)
        f3 = self.extractor3(f2)

        return f1, f2, f3

    def get_FGD_feat(self, x):
        return self.encoder(x)