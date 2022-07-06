import os
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save
from .config import JsonConfig
from .models import Glow
from .generator import Generator
import torch
import torch.nn as nn
from .feature_extractor import *
from scripts.cal_FGD import train_eval_FGD

class Trainer(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 data, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # set members
        # append date info
        self.log_dir = log_dir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        self.feature_extractor = Feature_Extractor(data_device, hparams)

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints
        self.use_noise = hparams.Train.use_noise

        # model relative
        self.graph = graph
        self.optim = optim

        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device

        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.train_dataset = data.get_train_dataset()
        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=True,
                                      drop_last=True)
                                      
        self.n_epoches = (hparams.Train.num_batches+len(self.data_loader)-1)
        self.n_epoches = self.n_epoches // len(self.data_loader)
        self.global_step = 0
        
        self.generator = Generator(data, data_device, log_dir, hparams)
        self.is_trinity = hparams.Dir.is_trinity

        # validation batch
        self.val_data_loader = DataLoader(data.get_validation_dataset(),
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      shuffle=False,
                                      drop_last=True)
        
        if not self.is_trinity:
            self.FGD_val_loader = DataLoader(data.get_FGD_val_dataset(),
                                        batch_size=500,
                                        num_workers=0,
                                        shuffle=False,
                                        drop_last=True)
            self.best_FGD = 1e6
            
        self.data = data
        
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Train.validation_log_gap
        self.plot_gaps = hparams.Train.plot_gap

        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
            
    def count_parameters(self, model):
         return sum(p.numel() for p in model.parameters() if p.requires_grad)    

    def prepare_cond(self, jt_data, ctrl_data):
        nn, seqlen, n_feats = jt_data.shape

        jt_data = jt_data.reshape((nn, -1))
        nn, seqlen, n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape(nn, -1)
        cond = torch.unsqueeze(torch.cat((jt_data, ctrl_data), axis=1), axis=-1)
        return cond.to(self.data_device)

    def train(self):

        self.global_step = self.loaded_step

        # begin to train
        for epoch in range(self.n_epoches):
            print("epoch", epoch)
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):

                # set to training state
                self.graph.train()
                
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                                                             
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                    
                # get batch data
                for k in batch:
                    if k == "style":
                        continue
                    batch[k] = batch[k].to(self.data_device)
                x = batch["x"]               # [batch_size, 27, len]  
                B, _, L = x.shape
                cond = batch["cond"]

                # init LSTM hidden
                if hasattr(self.graph, "module"):
                    self.graph.module.init_lstm_hidden()
                else:
                    self.graph.init_lstm_hidden()

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x=x[:self.batch_size // len(self.devices), ...],
                               cond=cond[:self.batch_size // len(self.devices), ...] if cond is not None else None, 
                               )
                    # re-init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                    else:
                        self.graph.init_lstm_hidden()
                
                #print("n_params: " + str(self.count_parameters(self.graph)))
                
                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                    
                
                # forward phase
                z, nll = self.graph(x=x, cond=cond) # z: [200, 27, 32], cond [200, 589, 32]
                                                    # control: [200, 42, 29]
                
                loss_generative = Glow.loss_generative(nll)

                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss_generative.backward()


                if i_batch % 5 == 0:
                    # init LSTM hidden
                    if hasattr(self.graph, "module"):
                        self.graph.module.init_lstm_hidden()
                    else:
                        self.graph.init_lstm_hidden()

                    x_recon = self.graph(z=None, cond=cond, eps_std=1.0, reverse=True)
                        
                    feat_criterion = nn.L1Loss()
                    feat1, _, _ = self.feature_extractor.get_feature(x.permute(0, 2, 1))
                    feat1_r, _, _ = self.feature_extractor.get_feature(x_recon.permute(0, 2, 1))
                    loss_perceptual = feat_criterion(feat1_r, feat1)

                    criterion = nn.L1Loss()
                    loss_recon = criterion(x_recon, x)

                    loss = loss_recon + 0.1 * loss_perceptual
                    loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                self.optim.step()

                if self.global_step % self.validation_log_gaps == 0:
                    # set to eval state
                    self.graph.eval()
                                        
                    # Validation forward phase
                    loss_val = 0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        for k in val_batch:
                            if k == "style":
                                continue
                            val_batch[k] = val_batch[k].to(self.data_device)
                            
                        with torch.no_grad():
                            
                            # init LSTM hidden
                            if hasattr(self.graph, "module"):
                                self.graph.module.init_lstm_hidden()
                            else:
                                self.graph.init_lstm_hidden()

                                
                            z_val, nll_val = self.graph(x=val_batch["x"], cond=val_batch["cond"], 
                            )
                            
                            # loss
                            loss_val = loss_val + Glow.loss_generative(nll_val)
                            n_batches = n_batches + 1        
                    
                    loss_val = loss_val/n_batches
                    self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)
                    
                                
                is_best = False
                if not self.is_trinity:
                    if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0 and (not self.is_trinity):
                        self.graph.eval()
                        FGD = train_eval_FGD(self.FGD_val_loader, self.graph, self.data, self.global_step)
                        if FGD < self.best_FGD:
                            is_best = True
                            self.best_FGD = FGD
                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=is_best,
                         max_checkpoints=self.max_checkpoints)


                # global step
                self.global_step += 1
            print(
                f'loss_generative: {loss_generative.item():.5f}, loss_recon: {loss_recon.item():.5f}, loss_perceptual: {loss_perceptual.item():.5f} / Validation Loss: {loss_val:.5f} '
            )

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
