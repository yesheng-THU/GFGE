# Audio-Driven Stylized Gesture Generation with Flow-Based Model (ECCV 2022)
![Teaser image](./teaser/teaser.png)

This is repository with training and inference code for paper [**"Audio-Driven Stylized Gesture Generation with Flow-Based Model"**].

## Requirements

* Linux OS
* NVIDIA GPUs. We tested on A100 GPUs.
* Python libraries: see [environment.yml](./environment.yml). You can use the following commands with Anaconda3 to create and activate your virtual environment:
  - `git clone https://github.com/yesheng-THU/GFGE.git`
  - `cd GFGE`
  - `conda env create -f environment.yml`
  - `conda activate GFGE`

## Getting started
### Datasets

In this work, we conducted our experiments on two datasets: TED Dataset and Trinity Dataset.

* For TED Dataset, you can download the raw data from [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/zeroyy_kaist_ac_kr/EYAPLf8Hvn9Oq9GMljHDTK4BRab7rl9hAOcnjkriqL8qSg) (16GB) and extract the ZIP file into `../ted_dataset`. Then you can use the following command to preprocess the TED Dataset:

  ```
  python data_processing/prepare_deepspeech_gesture_datasets.py
  ```

  The processed data will be under the folder `data/locomotion`. We also provide the [processed data](https://drive.google.com/file/d/18_mJ__wWAXZVSgkqCGC-NetOOd_bUgJ3/view?usp=sharing) for training the complete model and the [partial data](https://drive.google.com/file/d/1sdStqJ51X9TEF2MxNVjJneevqYsGfE1L/view?usp=sharing) for visualizing the latent space. You can directly download these NPZ files and place them under the folder `data/locomotion`.


* For Trinity Dataset, we used the [data](https://trinityspeechgesture.scss.tcd.ie/data/Trinity%20Speech-Gesture%20I/GENEA_Challenge_2020_data_release/) to train our models. Trinity College Dublin requires interested parties to sign a license agreement and receive approval before gaining access to this dataset. This is also the same data that was used for the [GENEA Challenge 2020]. Place the data under the `../trinity_dataset` folder and then run the following command:

  ```
  python data_processing/prepare_trinity_datasets.py
  ```

  The processed data will be under the folder `data/GENEA`.


### Feature Extractors

To successfully train and test our network, you also need to download some auxiliary files.
Feature extractors are required to compute the **Gesture Perceptual Loss**. You can either train your own feature extractors (by running `python scripts/train_gp_loss.py`) or directly download our [pretrained feature extractors](https://drive.google.com/file/d/1l0qy_IkwL42Aa7NB03cEVR9-sV_b8lp5/view?usp=sharing) and extract the ZIP file into `./feature_extractor`.

To calculate FGD metric during training and testing, you also need to download a [checkpoint](https://drive.google.com/file/d/1GPllMxd4mW_9e26upMJElhGC1AiHNIPg/view?usp=sharing) trained Human3.6M dataset (the same as [Yoon et al.](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context) proposed) and place it under the folder `./feature_extractor`.

### Model Checkpoints

### Usage



<!-- 训练Trinity的指令 python scripts/train.py hparams/preferred/trinity.json trinity
测试Trinity的指令 python scripts/test_trinity_sample.py
生成结果会在results/GENEA下

训练TED的指令 python scripts/train.py hparams/preferred/locomotion.json locomotion
测试TED的指令 python scripts/test_locomotion_sample.py
生成结果会在results/locomotion下 -->
