"""Train script.

Usage:
    train.py <hparams> <dataset>
"""
import os, sys
module_path = os.path.abspath(".")
if module_path not in sys.path:
    sys.path.append(module_path)

import motion
import datetime

from docopt import docopt
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    assert dataset in motion.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, motion.Datasets.keys()))
    assert os.path.exists(hparams), (
        "Failed to find hparams json `{}`".format(hparams))
    hparams = JsonConfig(hparams)
    dataset = motion.Datasets[dataset]
    
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")

	
    log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
    print("log_dir:" + str(log_dir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    is_training = hparams.Infer.pre_trained == ""

    if not hparams.Dir.is_trinity:
        data = dataset(hparams, is_training, log_path=log_dir)
    else:
        data = dataset(hparams, is_training)
    
    x_channels, cond_channels = data.n_channels()
    control_shape = data.get_control_shape()
    autoreg_shape = data.get_autoreg_shape()

    # build graph
    built = build(x_channels, cond_channels, control_shape, autoreg_shape, hparams, is_training)
    
    if is_training:

        # build trainer
        trainer = Trainer(**built, data=data, log_dir=log_dir, hparams=hparams)
        
        # train model
        trainer.train()

