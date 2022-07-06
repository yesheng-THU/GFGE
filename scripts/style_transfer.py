"""Train script.

Usage:
    style_transfer.py <hparams> <dataset>
"""
import os, sys
module_path = os.path.abspath(".")
if module_path not in sys.path:
    sys.path.append(module_path)

import motion
import numpy as np
import datetime
import subprocess
import pickle
from glow.builder import build
from glow.generator import Generator
from glow.config import JsonConfig
from scipy.interpolate import interp1d
import argparse
import scipy.io.wavfile as wav
import librosa
from textwrap import wrap
import matplotlib.animation as animation
import math
import time
import matplotlib.pyplot as plt

def extract_melspec1(X, fps, fs):
    mel_all = {}


    X = X.astype(float) / math.pow(2, 15)

    # assert fs % fps == 0

    hop_len = int(round(fs / fps))

    n_fft = int(fs * 0.13)
    C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=29, fmin=0.0, fmax=8000)
    C = np.log(C + 1e-6)
    print("fs: " + str(fs))
    # print("hop_len: " + str(hop_len))
    # print("n_fft: " + str(n_fft))
    print(C.shape)
    print(np.min(C), np.max(C))
    C_use = np.transpose(C)
    C_out = []
    for i in range(C_use.shape[0]):
        if np.sum(C_use[i]) != float("-inf") and np.sum(C_use[i]) != float("inf"):
            C_out.append(C_use[i])
    C_out = np.array(C_out)
    return C_out

def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


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
                          audio_path=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True):
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
        video_path = '{}.mp4'.format(save_path)
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


def resample_pose_seq(poses, duration_in_sec, fps):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_path', type=str, default='data/test',
                        help='directory where test audios are stored.')

    opt = parser.parse_args()

    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")

    fps = 30
    step = 1
    debug = True
    split = False

    if split:
        beach = fps * 10
    else:
        beach = fps * 60 * 60

    hparams_motion = "hparams/preferred/locomotion_test.json"  # args["<hparams>"]
    dataset_motion = "locomotion"  # args["<dataset>"]
    hparams_motion = JsonConfig(hparams_motion)
    dataset_motion = motion.Datasets[dataset_motion]

    log_dir = os.path.join(hparams_motion.Dir.log_root, "log_" + date)
    print("log_dir:" + str(log_dir))
    checkpoint_dir = hparams_motion.Infer.checkpoint_dir

    data_motion = dataset_motion(hparams_motion, True, log_path=checkpoint_dir)
    x_channels_motion, cond_channels_motion = data_motion.n_channels()

    built_motion = build(x_channels_motion, cond_channels_motion, None, None, hparams_motion, False)
    # Synthesize a lot of data.
    generator_motion = Generator(data_motion, built_motion['data_device'], log_dir, hparams_motion)

    bodypose, target_style, origin = generator_motion.style_transfer(built_motion['graph'], eps_std=1.0, style='MgnnQ2CN6yY')

    bodypose = inv_standardize(bodypose, data_motion.scaler)
    bodypose = bodypose.reshape(-1, 27)

    target_style = inv_standardize(target_style, data_motion.scaler)
    target_style = target_style.reshape(-1, 27)

    origin = inv_standardize(origin, data_motion.scaler)
    origin = origin.reshape(-1, 27)

    mean_data = np.array(
        [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039,
         -0.9236511, 0.3061306, -0.0012415, -0.5155854, 0.8129665, 0.0871897, 0.2348464, 0.1846561, 0.8091402,
         0.9271948, 0.2960011, -0.013189, 0.5233978, 0.8092403, 0.0725451, -0.2037076, 0.1924306, 0.8196916])
    
    create_video_and_save("./transfer", None, bodypose, mean_data, "test",
                          audio_path=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=False)
    
    create_video_and_save("./style", None, target_style, mean_data, "test",
                          audio_path=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=False)

    create_video_and_save("./origin", None, origin, mean_data, "test",
                          audio_path=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=False)

    print("")
