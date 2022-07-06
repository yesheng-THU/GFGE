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
### Dataset

In this work, we conducted our experiments on two datasets: TED Dataset and Trinity Dataset.

* For TED Dataset, you can download the raw data from [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/zeroyy_kaist_ac_kr/EYAPLf8Hvn9Oq9GMljHDTK4BRab7rl9hAOcnjkriqL8qSg) (16GB) and extract the ZIP file into `../ted_dataset`. Then you can use the following command to preprocess the TED Dataset.

  ```
  python data_processing/prepare_deepspeech_gesture_datasets.py
  ```

  The processed data will be under the folder `data/locomotion`. We also provide the processed data:  for training the complete model and for visualizing the latent space.

* For Trinity Dataset, 




<!-- 训练Trinity的指令 python scripts/train.py hparams/preferred/trinity.json trinity
测试Trinity的指令 python scripts/test_trinity_sample.py
生成结果会在results/GENEA下

训练TED的指令 python scripts/train.py hparams/preferred/locomotion.json locomotion
测试TED的指令 python scripts/test_locomotion_sample.py
生成结果会在results/locomotion下 -->
