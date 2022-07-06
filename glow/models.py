import torch
import torch.nn as nn
import numpy as np
from . import thops
from . import modules
from . import utils
from .modules import GRU
from .ae_model import Global_Encoder
import pickle


def build_vocab(cache_path, word_vec_path=None, feat_dim=None):
    with open(cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    if word_vec_path is None:
        lang_model.word_embedding_weights = None
    elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
        assert False

    return lang_model

def nan_throw(tensor, name="tensor"):
        stop = False
        if ((tensor!=tensor).any()):
            print(name + " has nans")
            stop = True
        if (torch.isinf(tensor).any()):
            print(name + " has infs")
            stop = True
        if stop:
            print(name + ": " + str(tensor))
            #raise ValueError(name + ' contains nans of infs')

def f(in_channels, out_channels, hidden_channels, cond_channels, 
        style_dim, network_model, num_layers, layer_idx, K):
    # in_channels : 13, out_channels : 28, hidden_channels : 512
    # cond_channels : 1149, network_model : FF/GRU/LSTM/Transformer, num_layers : 2
    if network_model=="Transformer":
        return modules.Transformer(out_channels, in_channels + cond_channels + 64, 2, hidden_channels, 2, 0.0)
    if network_model=="LSTM":
        return modules.LSTM(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    if network_model=="GRU":
        return modules.GRU(in_channels + cond_channels + 32, hidden_channels, out_channels, num_layers)
    if network_model=="FF":
        return nn.Sequential(
        nn.Linear(in_channels + cond_channels + 32 + 32, hidden_channels), nn.ReLU(inplace=False),
        nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=False),
        modules.LinearZeroInit(hidden_channels, out_channels))
    if network_model=="MIX":
        if layer_idx % 2 == 0:
            return modules.Transformer(out_channels, in_channels + cond_channels + 32 + 32, 2, hidden_channels, 2, 512, 0.0)
        else:
            return modules.LSTM(in_channels + cond_channels + 32 + 32, hidden_channels, out_channels, num_layers)


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    NetworkModel = ["LSTM", "GRU", "FF", "Transformer", "MIX"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels, cond_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 network_model="LSTM",
                 num_layers=2,
                 LU_decomposed=False,
                 style_channel=1356,
                 style_dim=32,
                 control_shape = 609,
                 autoreg_shape = 540,
                 layer_idx = 0,
                 K = 0,
                 use_noise = False
                 ):
                 
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert network_model in FlowStep.NetworkModel,\
            "network_model should be in `{}`".format(FlowStep.NetworkModel)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.network_model = network_model
        self.use_noise = use_noise
        
        if use_noise:
            self.noise_embed = GRU(in_channels, 256, in_channels//2+1, 1, 0.0)

        self.style_dim = style_dim
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels-in_channels // 2, hidden_channels, cond_channels,
                         style_dim, network_model, num_layers, layer_idx, K)
        elif flow_coupling == "affine":
            print("affine: in_channels = " + str(in_channels))
            self.f = f(in_channels // 2, 2*(in_channels-in_channels // 2), hidden_channels, cond_channels,
                         style_dim, network_model, num_layers, layer_idx, K)
            # in_channels : 27, hidden_channels : 512, cond_channels : 1149
            # network_model : FF/GRU/LSTM/Transformer, num_layers : 2
            print("Flowstep affine layer: " + str(in_channels))

    def init_lstm_hidden(self):
        if self.network_model == "LSTM" or self.network_model == "GRU" or self.network_model == "MIX":
            self.f.init_hidden()
        if self.use_noise:
            self.noise_embed.init_hidden()

    def forward(self, input, cond, logdet=None, reverse=False, feat=None, semantic=None, noise=None):
        if not reverse:
            return self.normal_flow(input, cond, logdet, feat=feat, semantic=semantic, noise=noise)
        else:
            return self.reverse_flow(input, cond, logdet, feat=feat, semantic=semantic, noise=noise)

    def normal_flow(self, input, cond, logdet, feat, semantic, noise):
        # input : [100, 27, 22], cond : [100, 1149, 22] 

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # z : [100, 27, 22], logdet : [100] --> [Batch]

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)
        # z : [100, 27, 22], logdet : [100] --> [Batch]

        # 3. coupling
        z1, z2 = thops.split_feature(z, "split") # z1 : [100, 13, 22], z2 : [100, 14, 22]
        z1_cond = torch.cat((z1, cond, feat), dim=1) # z1_cond : [100, 1149 + 64 * 3, 22]

        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1_cond)
        elif self.flow_coupling == "affine":        
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1) # h : [100, 28, 22]
            shift, scale = thops.split_feature(h, "cross") # shift : [100, 14, 22], scale : [100, 14, 22]
            scale = torch.sigmoid(scale + 2.) + 1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2]) + logdet
        

        # add noise term
        if self.use_noise:
            noise = self.noise_embed(noise) # [B, seq, 14]
            noise = noise.permute(0, 2, 1)
            z2 = z2 + noise

        z = thops.cat_feature(z1, z2)
        return z, cond, logdet

    def reverse_flow(self, input, cond, logdet, feat, semantic, noise):
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        z1_cond = torch.cat((z1, cond, feat), dim=1)

        if self.use_noise:
            noise = self.noise_embed(noise)
            noise = noise.permute(0, 2, 1)
            z2 = z2 - noise

        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = thops.split_feature(h, "cross")
            nan_throw(shift, "shift")
            nan_throw(scale, "scale")
            nan_throw(z2, "z2 unscaled")
            scale = torch.sigmoid(scale + 2.) + 1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2]) + logdet
            
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        nan_throw(z, "z permute_" + str(self.flow_permutation))
       # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, cond, logdet


class FlowNet(nn.Module):
    def __init__(self, x_channels, hidden_channels, cond_channels, K,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 network_model="LSTM",
                 num_layers=2,
                 LU_decomposed=False,
                 style_channel=1356,
                 control_shape = 609,
                 autoreg_shape = 540,
                 use_noise = False):
                 
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        N = cond_channels
        for idx in range(K):
            self.layers.append(
                FlowStep(in_channels=x_channels, # 27
                         hidden_channels=hidden_channels, # 512
                         cond_channels=N, # 1149
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         network_model=network_model,
                         num_layers=2,
                         LU_decomposed=LU_decomposed,
                         style_channel=style_channel,
                         style_dim=32,
                         control_shape=control_shape,
                         autoreg_shape=autoreg_shape,
                         layer_idx = idx,
                         K = K,
                         use_noise = use_noise))
            self.output_shapes.append(
                [-1, x_channels, 1])

    def init_lstm_hidden(self):
        for layer in self.layers:
            if isinstance(layer, FlowStep):                
                layer.init_lstm_hidden()

    def forward(self, z, cond, logdet=0., reverse=False, eps_std=None, feat=None, semantic=None, noise=None):
        if not reverse:
            for layer in self.layers:
                z, cond, logdet = layer(z, cond, logdet, reverse=False, feat=feat, semantic=semantic, noise=noise)
            return z, logdet
        else:
            for i,layer in enumerate(reversed(self.layers)):
                z, cond, logdet = layer(z, cond, logdet=0, reverse=True, feat=feat, semantic=semantic, noise=noise)
            return z


class Glow(nn.Module):

    def __init__(self, x_channels, cond_channels, control_shape, autoreg_shape, hparams):
        super().__init__()
        self.flow = FlowNet(x_channels=x_channels,
                            hidden_channels=hparams.Glow.hidden_channels,
                            cond_channels=cond_channels,
                            K=hparams.Glow.K,
                            actnorm_scale=hparams.Glow.actnorm_scale,
                            flow_permutation=hparams.Glow.flow_permutation,
                            flow_coupling=hparams.Glow.flow_coupling,
                            network_model=hparams.Glow.network_model,
                            num_layers=hparams.Glow.num_layers,
                            LU_decomposed=hparams.Glow.LU_decomposed,
                            style_channel=hparams.Data.style_channel,
                            control_shape = control_shape,
                            autoreg_shape = autoreg_shape,
                            use_noise = False)
        self.hparams = hparams
        self.is_trinity = hparams.Dir.is_trinity
        if self.is_trinity:
            self.audio_channels = 27
        else:
            self.audio_channels = 29
        self.pose_channels = x_channels
        self.text_channels = 300
        self.seqlen = hparams.Data.seqlen
        self.lookahead = hparams.Data.n_lookahead
        self.pose_autoreg_shape = self.pose_channels * self.seqlen
        self.audio_autoreg_shape = self.audio_channels * (self.seqlen + 1)
        self.text_autoreg_shape = self.text_channels * (self.seqlen + 1)
        self.global_encoder = Global_Encoder(self.pose_channels, self.audio_channels)

        # register prior hidden
        num_device = len(utils.get_proper_device(hparams.Device.glow, False))
        assert hparams.Train.batch_size % num_device == 0
        self.batch = hparams.Train.batch_size
        if hparams.Train.is_train:
            seqlen = hparams.Data.seqlen
            lookahead = hparams.Data.n_lookahead
            if self.is_trinity:
                self.z_shape = [hparams.Train.batch_size // num_device, x_channels, 120-seqlen-lookahead]
            else:
                self.z_shape = [hparams.Train.batch_size // num_device, x_channels, 42-seqlen-lookahead]
        else:
            self.z_shape = [hparams.Train.batch_size // num_device, x_channels, 1]
        if hparams.Glow.distribution == "normal":
            self.distribution = modules.GaussianDiag()
        elif hparams.Glow.distribution == "studentT":
            self.distribution = modules.StudentT(hparams.Glow.distribution_param, x_channels)

    def init_lstm_hidden(self):
        self.flow.init_lstm_hidden()

    def forward(self, x=None, cond=None, z=None,
                eps_std=None, reverse=False, noise=None, z_shape=None):

        pose = cond[:,:self.pose_autoreg_shape,:].permute(0, 2, 1) # [B, L, 12*27]
        audio = cond[:,self.pose_autoreg_shape:,:].permute(0, 2, 1) # [B, L, 13*29]
        pose = pose.reshape(pose.shape[0], pose.shape[1], self.seqlen, self.pose_channels)
        audio = audio.reshape(audio.shape[0], audio.shape[1], (self.seqlen+self.lookahead+1), self.audio_channels)

        feat = self.global_encoder(pose, audio) # [B, L, 32]
        feat = feat.permute(0, 2, 1) # [B, 32, L]
        semantic = None
        if not reverse:
            return self.normal_flow(x, cond, feat, semantic, noise)
        else:
            return self.reverse_flow(z, cond, eps_std, feat, semantic, noise, z_shape)
    

    def normal_flow(self, x, cond, feat, semantic, noise):
    
        n_timesteps = thops.timesteps(x)

        logdet = torch.zeros_like(x[:, 0, 0]) # [batch_size]

        # encode
        z, objective = self.flow(x, cond, logdet=logdet, reverse=False, feat=feat, semantic=semantic, noise=noise)

        # prior
        objective += self.distribution.logp(z) # to maximize this object

        # return
        nll = (-objective) / float(np.log(2.) * n_timesteps) # negative log likelihood
        return z, nll

    def reverse_flow(self, z, cond, eps_std, feat, semantic, noise, z_shape):
        if z_shape is None:
            z_shape = self.z_shape
        if z is None:
            z = self.distribution.sample(z_shape, eps_std, device=cond.device)

        x = self.flow(z, cond, eps_std=eps_std, reverse=True, feat=feat, semantic=semantic, noise=noise)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)
        