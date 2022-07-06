import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import scipy.io.wavfile as wav
import math
import librosa
import matplotlib
from tqdm import tqdm
import tikzplotlib
from .config import JsonConfig
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import lmdb
import random
import pyarrow
from sklearn.preprocessing import normalize
from scipy.interpolate import interp1d
import soundfile as sf

class Generator(object):
    def __init__(self, data, data_device, log_dir, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # model relative
        self.data_device = data_device
        self.seqlen = hparams.Data.seqlen
        self.n_lookahead = hparams.Data.n_lookahead
        self.data = data
        self.log_dir = log_dir
        self.tsne = None
        self.pca = None

        # test batch
        self.test_data_loader = DataLoader(data.get_test_dataset(),
                                           batch_size=hparams.Train.batch_size,
                                           num_workers=1,
                                           shuffle=False,
                                           drop_last=True)
        self.train_data_loader = DataLoader(data.get_train_dataset(),
                                           batch_size=100,
                                           num_workers=1,
                                           shuffle=False,
                                           drop_last=True)
        
        self.test_batch = next(iter(self.test_data_loader))
        for k in self.test_batch:
            if k == "style":
                continue
            self.test_batch[k] = self.test_batch[k].to(self.data_device)

    def resample_pose_seq(self, poses, duration_in_sec, fps):
        n = len(poses)
        x = np.arange(0, n)
        y = poses
        f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
        expected_n = duration_in_sec * fps
        x_new = np.arange(0, n, n / expected_n)
        interpolated_y = f(x_new)
        if hasattr(poses, 'dtype'):
            interpolated_y = interpolated_y.astype(poses.dtype)
        return interpolated_y

    def convert_pose_seq_to_dir_vec(self, pose):
        dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]
        
        if pose.shape[-1] != 3:
            pose = pose.reshape(pose.shape[:-1] + (-1, 3))

        if len(pose.shape) == 3:
            dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
            for i, pair in enumerate(dir_vec_pairs):
                dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
                dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
        elif len(pose.shape) == 4:  # (batch, seq, ...)
            dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
            for i, pair in enumerate(dir_vec_pairs):
                dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
            for j in range(dir_vec.shape[0]):  # batch
                for i in range(len(dir_vec_pairs)):
                    dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
        else:
            assert False

        return dir_vec


    def calc_tsne(self, raw):
        if self.tsne is None:
            self.tsne = TSNE(n_components=2, init='pca', random_state=7)  # n_iter = xxx
        result = self.tsne.fit_transform(raw)
        return result
    
    def calc_pca(self, raw):
        if self.pca is None:
            self.pca = PCA(n_components=2)
        return self.pca.fit_transform(raw)

    def distinct_labels_and_indices(self, labels):
        distinct_labels = list(set(labels))
        distinct_labels.sort()
        num_labels = len(distinct_labels)
        indices_i = {label: [] for label in distinct_labels}
        for i, label in enumerate(labels):
            indices_i[label].append(i)
        indices_i = {label: np.array(indices) for label, indices in indices_i.items()}
        return num_labels, distinct_labels, indices_i

    def plot2D(self, data, labels, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
        # data = (data - x_min) / (x_max - x_min)

        fig, ax = plt.subplots(figsize=(10, 8))
        cjet = cm.get_cmap("jet")

        num_labels, distinct_labels, indices = self.distinct_labels_and_indices(labels)

        for i, label in enumerate(distinct_labels):
            index = indices[label]
            ax.scatter(data[index, 0], data[index, 1], label=label, c=[cjet(1.0 * i / num_labels)], linewidths=0.)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0, 1, 1))

        fig.tight_layout()
        tikzplotlib.save("%s/plot2d.tex" % dir, figure=fig, strict=True)
        plt.savefig("%s/plot2d.png" % dir)

        return fig
    
    def cal_dis(self, data, labels, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        num_labels, distinct_labels, indices = self.distinct_labels_and_indices(labels)
        z_all = []
        labels = []
        nums = []

        for i, label in enumerate(distinct_labels):
            index = indices[label] # [num_label]
            z = data[index] # [num_label, 27]
            nums.append(z.shape[0])
            z = z.mean(axis=0) # [27]
            z_all.append(z)
            labels.append(label)

        dist = []
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                d = np.sum((z_all[i] - z_all[j]) ** 2)
                dist.append((labels[i], labels[j], d, nums[i], nums[j]))
            
            print(i)

        dist = sorted(dist, key = lambda x:x[2])
        with open("dist.txt", "w") as f:
            for i in dist:
                (a, b, c, d, e) = i
                f.write(str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + ' ' + str(e))
                f.write("\n")
        
    def prepare_cond(self, jt_data, ctrl_data):
        nn, seqlen, n_feats = jt_data.shape

        jt_data = jt_data.reshape((nn, seqlen * n_feats))
        nn, seqlen, n_feats = ctrl_data.shape
        ctrl_data = ctrl_data.reshape((nn, seqlen * n_feats))
        cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data, ctrl_data), axis=1), axis=-1))
        return cond.to(self.data_device)


    def generate_sample(self, graph, eps_std=1.0, step=0, counter=0):
        print("generate_sample")
        graph = graph.eval()

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy() # [1, 42, 27]
        control_all = batch["control"].cpu().numpy() # [1, 42, 29]

        seqlen = self.seqlen
        n_lookahead = self.n_lookahead

        clip_length = 1
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()

        nn, n_timesteps, n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps - n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32)  # initialize from a mean pose
        autoreg[:, :seqlen, :] = autoreg_all[:, :seqlen, :]
        sampled_all[:, :seqlen, :] = autoreg

        # Loop through control sequence and generate new data
        with torch.no_grad():
            for i in range(0, control_all.shape[1] - seqlen - n_lookahead, clip_length):
                control = control_all[:, i:(i + seqlen + n_lookahead + clip_length), :]

                # prepare conditioning for moglow (control + previous poses)
                cond = self.prepare_cond(autoreg.copy(), control.copy())


                # sample from Moglow
                sampled = graph(z=None, cond=cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:, :, 0].reshape(sampled.shape[0], clip_length, -1)

                # store the sampled frame
                sampled_all[:, (i + seqlen): (i + seqlen + clip_length), :] = sampled

                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:, clip_length:, :].copy(), sampled[:, :, :]), axis=1)

        return sampled_all[:, :, :], autoreg_all[:, :, :]
    
    def trinity_sample(self, graph, eps_std=1.0):
        print("generate_sample")
        graph = graph.eval()

        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy() # [1, 42, 27]
        control_all = batch["control"].cpu().numpy() # [1, 42, 29]

        seqlen = self.seqlen
        n_lookahead = self.n_lookahead

        clip_length = 1
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()

        nn, n_timesteps, n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps - n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32)  # initialize from a mean pose
        autoreg[:, :seqlen, :] = autoreg_all[:, :seqlen, :]
        sampled_all[:, :seqlen, :] = autoreg

        # Loop through control sequence and generate new data
        with torch.no_grad():
            for i in range(0, control_all.shape[1] - seqlen - n_lookahead, clip_length):
                control = control_all[:, i:(i + seqlen + n_lookahead + clip_length), :]

                # prepare conditioning for moglow (control + previous poses)
                cond = self.prepare_cond(autoreg.copy(), control.copy()) 

                # sample from Moglow
                sampled = graph(z=None, cond=cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:, :, 0].reshape(sampled.shape[0], clip_length, -1)

                # store the sampled frame
                sampled_all[:, (i + seqlen): (i + seqlen + clip_length), :] = sampled

                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:, clip_length:, :].copy(), sampled[:, :, :]), axis=1)

        return sampled_all[:, :, :], autoreg_all[:, :-n_lookahead, :]

    def extract_melspec1(self, X, fps, fs):
        mel_all = {}


        X = X.astype(float) / math.pow(2, 15)

        # assert fs % fps == 0

        hop_len = int(round(fs / fps))

        n_fft = int(fs * 0.13)
        C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=29, fmin=0.0, fmax=8000)
        C = np.log(C + 1e-6)
        print("fs: " + str(fs))
        print(C.shape)
        print(np.min(C), np.max(C))
        C_use = np.transpose(C)
        C_out = []
        for i in range(C_use.shape[0]):
            if np.sum(C_use[i]) != float("-inf") and np.sum(C_use[i]) != float("inf"):
                C_out.append(C_use[i])
        C_out = np.array(C_out)
        return C_out

    def style_transfer(self, graph, eps_std=1.0, style='AYzA2uyd9_s'):
        print("style transfer")

        graph = graph.eval()

        progress = tqdm(self.train_data_loader)
        idx = -1
        for i, batch in enumerate(progress):
            batch_style = batch['style']
            if style in batch_style:
                idx = batch_style.index(style)
                break
        
        with torch.no_grad():
            x, cond = batch['x'][idx].unsqueeze(0), batch['cond'][idx].unsqueeze(0)
            x, cond = x.to(self.data_device), cond.to(self.data_device)
            seed0 = batch['seed'][idx].unsqueeze(0).to(self.data_device)
            z, nll = graph(x=x, cond=cond) # z : [1, 27, 32]
        
        batch = self.test_batch

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        seqlen = self.seqlen
        n_lookahead = self.n_lookahead

        clip_length = 1
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()

        nn, n_timesteps, n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps - n_lookahead, n_feats))
        autoreg = seed0.permute(0, 2, 1).cpu().numpy()  # initialize from style pose
        
        dropout = 0.5
        mask = np.random.rand(nn, seqlen, n_feats) < (1 - dropout)
        autoreg = autoreg * mask

        sampled_all[:, :seqlen, :] = autoreg

        # Loop through control sequence and generate new data
        with torch.no_grad():
            for i in range(0, control_all.shape[1] - seqlen - n_lookahead, clip_length):
                control = control_all[:, i:(i + seqlen + n_lookahead + clip_length), :]
                z_style = z[:,:,i].unsqueeze(2)

                # prepare conditioning for moglow (control + previous poses)
                cond = self.prepare_cond(autoreg.copy(), control.copy())

                # sample from Moglow
                sampled = graph(z=z_style, cond=cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:, :, 0].reshape(sampled.shape[0], clip_length, -1)

                # store the sampled frame
                sampled_all[:, (i + seqlen): (i + seqlen + clip_length), :] = sampled

                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:, clip_length:, :].copy(), sampled[:, :, :]), axis=1)

        return sampled_all[:, seqlen:, :], x.transpose(1, 2).cpu().numpy(), autoreg_all[:, seqlen:, :]


    def generate_test_sample(self, graph, eps_std=1.0, step=0, counter=0):
        print("generate_test_sample")
        # graph.eval()
        test_data_path = '../ted_dataset/lmdb_test'
        lmdb_env = lmdb.open(test_data_path, readonly=True, lock=False)

        with lmdb_env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]
            key = random.choice(keys)
            print('key', key)

            buf = txn.get(key)
            video = pyarrow.deserialize(buf)
            vid = video['vid']
            clips = video['clips']

            # select clip
            n_clips = len(clips)
            clip_idx = random.randrange(n_clips)
            # clip_idx = 14
            print('clip_idx', clip_idx)

            clip_poses = clips[clip_idx]['skeletons_3d']
            clip_audio = clips[clip_idx]['audio_raw']
            clip_words = clips[clip_idx]['words']
            clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]

        clip_audio = clip_audio.astype(np.float32)
        clip_len = clip_audio.shape[0]
        clip_audio[clip_len // 2: ] = 0

        sr = 16000
        audio_path = 'sample.wav'
        sf.write(audio_path, clip_audio, sr)

        print([x[0] for x in clip_words])

        control_all = self.extract_melspec1(clip_audio, 15, 16000).astype('float32')
        control_all = control_all.reshape(1, -1, 29)[:, :, :]
        control_all = self.standardize(data=control_all, scaler=self.data.a_scaler)


        clip_poses = self.resample_pose_seq(clip_poses, clip_time[1] - clip_time[0], 15)
        autoreg_all = self.convert_pose_seq_to_dir_vec(clip_poses)
        autoreg_all = autoreg_all.reshape(1, -1, 27)[:, :, :]
        mean_data = np.array(
            [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039,
            -0.9236511, 0.3061306, -0.0012415, -0.5155854, 0.8129665, 0.0871897, 0.2348464, 0.1846561, 0.8091402,
            0.9271948, 0.2960011, -0.013189, 0.5233978, 0.8092403, 0.0725451, -0.2037076, 0.1924306, 0.8196916])
        autoreg_all -= mean_data
        autoreg_all = self.standardize(data=autoreg_all, scaler=self.data.scaler)

        seqlen = self.seqlen
        n_lookahead = self.n_lookahead

        clip_length = 1
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()

        nn, n_timesteps, n_feats = control_all.shape
        n_feats = 27
        sampled_all = np.zeros((nn, n_timesteps - n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32)  # initialize from a mean pose
        autoreg[:, :seqlen, :] = autoreg_all[:, :seqlen, :]
        sampled_all[:, :seqlen, :] = autoreg

        # Loop through control sequence and generate new data
        with torch.no_grad():
            for i in range(0, control_all.shape[1] - seqlen - n_lookahead, clip_length):
                control = control_all[:, i:(i + seqlen + n_lookahead + clip_length), :]

                # prepare conditioning for moglow (control + previous poses)
                cond = self.prepare_cond(autoreg.copy(), control.copy())

                # sample from Moglow
                sampled = graph(z=None, cond=cond, eps_std=eps_std, reverse=True)
                sampled = sampled.cpu().numpy()[:, :, 0].reshape(sampled.shape[0], clip_length, -1)

                # store the sampled frame
                sampled_all[:, (i + seqlen): (i + seqlen + clip_length), :] = sampled

                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:, clip_length:, :].copy(), sampled[:, :, :]), axis=1)
       
        return sampled_all[:, :, :], autoreg_all[:, :, :]


    def standardize(self, data, scaler):
        shape = data.shape
        flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
        scaled = scaler.transform(flat).reshape(shape)
        return scaled

    def smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def generate_code(self, graph):
        graph = graph.eval()
        progress = tqdm(self.train_data_loader)
        style_all = []
        z_all = []

        specific = ["XY_lzonfE3I", "fHfhorJnAEI", "QoTSdOkjEVs", "rufeS-lZJg8", "E_lb3D7Ay-M"]
        # specific = ['UxLRv0FEndM', 'Erm4vP1vTn8', 'qtcWebAYmKY', 'C_eFjLZqXt8', 'qYvXk_bqlBk']

        for i_batch, batch in enumerate(progress):
            with torch.no_grad():
                # Initialize the lstm hidden state
                if hasattr(graph, "module"):
                    graph.module.init_lstm_hidden()
                else:
                    graph.init_lstm_hidden()

                z_val, _ = graph(x=batch["x"].to(self.data_device), cond=batch["cond"].to(self.data_device))
                
                style_labels = batch["style"] # [100]
                z_val = z_val.transpose(1, 2).cpu().numpy() # [100, 30, 27]
                for i in range(len(style_labels)):
                    if style_labels[i] not in specific:
                        continue
                    z = z_val[i, :, :] # [30, 27]
                    z = z.mean(axis=0) # [27] latent_code
                    z_all.append(z)
                    style_all.append(style_labels[i])

        outputfig = self.log_dir
        z_all = np.array(z_all)
        print('total num: ', len(style_all))
        print('style num: ' ,len(set(style_all)))
        tsne_code = self.calc_tsne(z_all)
        self.plot2D(tsne_code, style_all, outputfig)
        from sklearn.cluster import KMeans
        from sklearn import metrics
        kmean_model = KMeans(n_clusters=5).fit(tsne_code)
        labels = kmean_model.labels_
        print('Scoeff:', metrics.silhouette_score(tsne_code, labels, metric="euclidean"))
        print('CHI:', metrics.calinski_harabasz_score(tsne_code, labels))

        return tsne_code, style_all
