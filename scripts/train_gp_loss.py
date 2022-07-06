import os, sys
module_path = os.path.abspath(".")
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from glow.config import JsonConfig
import motion
from torch.utils.data import DataLoader
from motion import *
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time, subprocess
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from textwrap import wrap
from glow.vae_model import *

# Device configuration
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def convert_dir_vec_to_pose(vec):
    dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                     (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def create_video_and_save(save_path, target, output, mean_data, title,
                          audio_path=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True, epoch=0):
    dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                     (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
    print('rendering a video...')
    start = time.time()

    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # un-normalization and convert to poses
    mean_data = mean_data.flatten()
    output = output + mean_data
    output_poses = convert_dir_vec_to_pose(output)
    target_poses = None
    if target is not None:
        target = target + mean_data
        target_poses = convert_dir_vec_to_pose(target)

    def animate(i):
        for k, name in enumerate(['human', 'generated']):
            if name == 'human' and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == 'generated' and i < len(output):
                pose = output_poses[i]
            else:
                pose = None

            if pose is not None:
                axes[k].clear()
                for j, pair in enumerate(dir_vec_pairs):
                    axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                 [pose[pair[0], 2], pose[pair[1], 2]],
                                 [pose[pair[0], 1], pose[pair[1], 1]],
                                 zdir='z', linewidth=5)
                axes[k].set_xlim3d(-0.5, 0.5)
                axes[k].set_ylim3d(0.5, -0.5)
                axes[k].set_zlim3d(0.5, -0.5)
                axes[k].set_xlabel('x')
                axes[k].set_ylabel('z')
                axes[k].set_zlabel('y')
                axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(output)))

    if target is not None:
        num_frames = max(len(target), len(output))
    else:
        num_frames = len(output)
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    try:
        video_path = '{}/{}.mp4'.format(save_path, title, epoch)
        ani.save(video_path, fps=15, dpi=150)  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, 'RuntimeError'

    # merge audio and video
    if audio_path is not None:
        merged_video_path = '{}/temp_with_audio.mp4'.format(save_path)
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, '-shortest')
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    return output_poses, target_poses


def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


if __name__ == '__main__':
    # Hyper-parameters
    pose_dim = 27
    n_pose = 30
    num_epochs = 200
    batch_size = 128
    learning_rate = 0.0005
    loader_workers = 4
    plot_interval = 1000
    save_interval = 20
    pose_diff = True

    hparams_motion = "hparams/preferred/locomotion_test.json"  # args["<hparams>"]
    dataset_motion = "locomotion"  # args["<dataset>"]
    hparams_motion = JsonConfig(hparams_motion)
    dataset_motion = Datasets[dataset_motion]

    log_dir = os.path.join(hparams_motion.Dir.log_root, "locomotion_feature")
    print("log_dir:" + str(log_dir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data_motion = dataset_motion(hparams_motion, True, log_dir)
    train_data_loader = DataLoader(data_motion.get_train_dataset(),
                                           batch_size=batch_size,
                                           num_workers=loader_workers,
                                           shuffle=True,
                                           drop_last=True)


model = Autoencoder(n_pose, pose_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Start training
for epoch in range(num_epochs):
    progress = tqdm(train_data_loader)
    for i, batch in enumerate(progress):
        model.train()
        # Forward pass
        for k in batch:
            if k == "style":
                continue
            batch[k] = batch[k].to(device)
        x = batch["x"].permute(0, 2, 1)
        x_recon = model(x, n_pose)
        
        # Compute reconstruction loss
        recon_loss = F.l1_loss(x_recon, x, reduction='none')
        recon_loss = torch.mean(recon_loss, dim=(1, 2))
        if pose_diff:
            target_diff = x[:, 1:] - x[:, :-1]
            recon_diff = x_recon[:, 1:] - x_recon[:, :-1]
            recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))
        recon_loss = torch.sum(recon_loss)
        
        loss = recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % plot_interval == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Recon_Loss: {:.4f}, KL_Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(train_data_loader), recon_loss.item(), 0.0))
    
    if (epoch + 1) % save_interval == 0:
        torch.save(model, os.path.join(log_dir, f'ae_{epoch+1}.pth'))
        with torch.no_grad():
            # Save the reconstruct motion sequence
            model.eval()
            out, x = x_recon[0:1,:,:], x[0:1,:,:] # [1, 30, 27]
            out, x = out.cpu().numpy(), x.cpu().numpy()
            bodypose = inv_standardize(out, data_motion.scaler)
            target = inv_standardize(x, data_motion.scaler)
            bodypose = bodypose.reshape(-1, 27)
            target = target.reshape(-1, 27)
            mean_data = np.array(
                [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039,
                -0.9236511, 0.3061306, -0.0012415, -0.5155854, 0.8129665, 0.0871897, 0.2348464, 0.1846561, 0.8091402,
                0.9271948, 0.2960011, -0.013189, 0.5233978, 0.8092403, 0.0725451, -0.2037076, 0.1924306, 0.8196916])
            create_video_and_save(log_dir, target, bodypose, mean_data, "test_{}".format(str(epoch+1)),
                                audio_path=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=False)
            print("save sample ...")
