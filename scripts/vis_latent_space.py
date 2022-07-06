import os, sys
module_path = os.path.abspath(".")
if module_path not in sys.path:
    sys.path.append(module_path)

import datetime
from glow.config import JsonConfig
import motion
from glow.builder import build
from glow.generator import Generator

if __name__ == "__main__":

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

    hparams_motion = "hparams/preferred/locomotion_latent_code.json"  # args["<hparams>"]
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
    bodypose, _ = generator_motion.generate_code(built_motion['graph'])
    
    print("")
