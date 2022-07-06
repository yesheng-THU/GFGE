"""Train script.

Usage:
    test_sample.py <hparams> <dataset>
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


if __name__ == "__main__":

    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")

    hparams_motion = "hparams/preferred/trinity_test.json"  # args["<hparams>"]
    dataset_motion = "trinity"  # args["<dataset>"]
    hparams_motion = JsonConfig(hparams_motion)
    dataset_motion = motion.Datasets[dataset_motion]

    log_dir = os.path.join(hparams_motion.Dir.log_root, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("log_dir:" + str(log_dir))

    data_motion = dataset_motion(hparams_motion, False)
    x_channels_motion, cond_channels_motion = data_motion.n_channels()

    built_motion = build(x_channels_motion, cond_channels_motion, None, None, hparams_motion, False)
    # Synthesize a lot of data.
    generator_motion = Generator(data_motion, built_motion['data_device'], log_dir, hparams_motion)

    bodypose, target = generator_motion.trinity_sample(built_motion['graph'], eps_std=1.0)
    data_motion.save_animation(None, bodypose, os.path.join(log_dir, 'fake'))
    data_motion.save_animation(None, target, os.path.join(log_dir, 'real'))
    print("")
