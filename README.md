<div align="center">
  <p>
    <a align="center" target="_blank">
      <img width="100%" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/banner/PIDNet_banner.png"></a>
  </p>
</div>

# <div align="center">NetsPresso tutorial for PIDNet compression</div>
## Order of the tutorial
[0. Sign up](#0-sign-up) </br>
[1. Install](#1-install) </br>
[2. Prepare the dataset](#2-prepare-the-dataset) </br>
[3. Training](#3-training) </br>
[4. Convert PIDNet to _torchfx.pt](#4-convert-pidnet-to-_torchfxpt) </br>
[5. Model compression with NetsPresso Python Package](#5-model-compression-with-netspresso-python-package)</br>
[6. Fine-tuning the compressed model](#6-fine-tuning-the-compressed-model)</br>
[7. Evaluation](#7-evaluation)</br>
[8. Custom inputs](#8-custom-inputs)</br>

## 0. Sign up
A NetsPresso account is required to use the NetsPresso Python Package. If you don't have a NetsPresso account, please sign up first.
You can sign up here: https://netspresso.ai/signup
</br>

## 1. Install
Clone repo, including
[**PyTorch >= 1.11, < 2.0**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Nota-NetsPresso/PIDNet_nota  # clone
```
</br>

## 2. Prepare the dataset
* Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets and unzip them in `data/cityscapes` and `data/camvid` dirs.
* Check if the paths contained in lists of `data/list` are correct for dataset images.

#### :smiley_cat: Instruction for preparation of CamVid data (remains discussion) :smiley_cat:

* Download the images and annotations from [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid), where the resolution of images is 960x720 (original);
* Unzip the data and put all the images and all the colored labels into `data/camvid/images/` and `data/camvid/labels`, respectively;
* Following the split of train, val and test sets used in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial), we have generated the dataset lists in `data/list/camvid/`;
* Finished!!! (We have open an issue for everyone who's interested in CamVid to discuss where to download the data and if the split in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial) is correct. BTW, do not directly use the split in [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid), which is wrong and will lead to unnormal high accuracy. We have revised the CamVid content in the paper and you will see the correct results after its announcement.)
</br>

## 3. Training
* Download the ImageNet pretrained models and put them into `pretrained_models/imagenet/` dir.
* For example, train the PIDNet-S on Cityscapes with batch size of 12 on 2 GPUs:
````bash
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml GPUS (0,1) TRAIN.BATCH_SIZE_PER_GPU 6
````
* Or train the PIDNet-L on Cityscapes using train and val sets simultaneously with batch size of 12 on 4 GPUs:
````bash
python tools/train.py --cfg configs/cityscapes/pidnet_large_cityscapes_trainval.yaml GPUS (0,1,2,3) TRAIN.BATCH_SIZE_PER_GPU 3
````
</br>

## 4. Convert PIDNet to _torchfx.pt
* Download the finetuned models for Cityscapes and CamVid and put them into `pretrained_models/cityscapes/` and `pretrained_models/camvid/` dirs, respectively.
* For example, convert the PIDNet-S on Cityscapes val set:
````bash
python tools/export_netspresso.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt
````
Executing this code will create 'model_modelfx.pt' and 'model_headfx.pt'.<br/>

## 5. Model compression with NetsPresso Python Package<br/>
Upload & compress your 'model_modelfx.pt' by using NetsPresso Python Package
### 5_1. Install NetsPresso Python Package
```bash
pip install netspresso
```
### 5_2. Upload & compress
First, import the packages and set a NetsPresso username and password.
```python
from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)
```
Second, upload 'model_modelfx.pt', which is the model converted to torchfx in step 4, with the following code.
```python
# Upload Model
UPLOAD_MODEL_NAME = "pidnet_model"
TASK = Task.SEMANTIC_SEGMENTATION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = "./model_modelfx.pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [1024, 1024]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)
```
Finally, you can compress the uploaded model with the desired options through the following code.
```python
# Recommendation Compression
COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.6
OUTPUT_PATH = "./compressed_pidnet.pt"
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    output_path=OUTPUT_PATH,
)
```

<details>
<summary>Click to check 'Full upload & compress code'</summary>

```bash
pip install netspresso
```

```python
from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

# Upload Model
UPLOAD_MODEL_NAME = "pidnet_model"
TASK = Task.SEMANTIC_SEGMENTATION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = "./model_modelfx.pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [1024, 1024]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_layers=INPUT_SHAPES,
)

# Recommendation Compression
COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.6
OUTPUT_PATH = "./compressed_pidnet.pt"
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    output_path=OUTPUT_PATH,
)
```

</details>

More commands can be found in the official NetsPresso Python Package docs: https://nota-netspresso.github.io/netspresso-python-docs/build/html/index.html <br/>

Alternatively, you can do the same as above through the GUI on our website: https://console.netspresso.ai/models<br/><br/>

## 6. Fine-tuning the compressed model</br>
After compression, retraining is necessary. You can retrain with the following code.<br>
Along with the --netspresso option, you need to put the path of the compressed model in the --model option and the path of model_headfx.pt that came out while converting to torchfx in the --head option.
```bash
python tools/train.py --netspresso --model model_model.pt --head model_head.pt --cfg configs/cityscapes/pidnet_small_cityscapes.yaml
```
If you want to perform additional compression, compress x_model_model_pt from training as in Step 5.
In the above command, put the path of the newly compressed model in the --model option. In the --head option, you need to change it to the path of x_model_head_pt that came out through retraining.
<br>

## 7. Evaluation<br/>
After retraining, files like x_model_model.pt and x_model_head.pt come out. You can run the model through eval.py.
````bash
python tools/eval.py --netspresso --model x_model_model.pt --head x_model_head.pt
````

## 8. Custom inputs<br/>
Also, you can put all your images in `samples/` and then run the command below using compressed and retrained model for image format of .png:
````bash
python tools/custom.py --netspresso --model x_model_model.pt --t '.png'
````

You can use the compressed model however you like! </br></br>

## <div align="center">Contact</div>

Join our <a href="https://github.com/orgs/Nota-NetsPresso/discussions">Discussion Forum</a> for providing feedback or sharing your use cases, and if you want to talk more with Nota, please contact us <a href="https://www.nota.ai/contact-us">here</a>.</br>
Or you can also do it via email(contact@nota.ai) or phone(+82 2-555-8659)!

<br>
<div align="center">
  <a href="https://github.com/Nota-NetsPresso" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/github_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/github.png">
      <img alt="github" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/github.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.facebook.com/NotaAI" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/facebook_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/facebook.png">
      <img alt="facebook" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/facebook.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/nota_ai" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/twitter_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/twitter.png">
      <img alt="twitter" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/twitter.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.youtube.com/channel/UCeewYFAqb2EqwEXZCfH9DVQ" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/youtube_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/youtube.png">
      <img alt="youtube" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/youtube.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/nota-incorporated" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/linkedin_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/linkedin.png">
      <img alt="youtube" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/linkedin.png" width="3%">
    </picture>
  </a>
</div>

</br>
</br>
</br>

# PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pidnet-a-real-time-semantic-segmentation/real-time-semantic-segmentation-on-camvid)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-camvid?p=pidnet-a-real-time-semantic-segmentation) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pidnet-a-real-time-semantic-segmentation/real-time-semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-cityscapes?p=pidnet-a-real-time-semantic-segmentation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pidnet-a-real-time-semantic-segmentation/real-time-semantic-segmentation-on-cityscapes-1)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-cityscapes-1?p=pidnet-a-real-time-semantic-segmentation) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official repository for our recent work: PIDNet ([PDF](https://arxiv.org/pdf/2206.02066)）

## Highlights
<p align="center">
  <img src="figs/cityscapes_score.jpg" alt="overview-of-our-method" width="500"/></br>
  <span align="center">Comparison of inference speed and accuracy for real-time models on test set of Cityscapes.</span> 
</p>

* **Towards Real-time Applications**: PIDNet could be directly used for the real-time applications, such as autonomous vehicle and medical imaging.
* **A Novel Three-branch Network**: Addtional boundary branch is introduced to two-branch network to mimic the PID controller architecture and remedy the overshoot issue of previous models.
* **More Accurate and Faster**: PIDNet-S presents 78.6% mIOU with speed of 93.2 FPS on Cityscapes test set and 80.1% mIOU with speed of 153.7 FPS on CamVid test set. Also, PIDNet-L becomes the most accurate one (80.6% mIOU) among all the real-time networks for Cityscapes.

## Updates
   - This paper was accepted by CVPR 2023, new version and associated materials will be available soon! (Apr/06/2023)
   - Fixed the data bug for Camvid and the new version of arXiv preprint will be available on Jun 13th. (Jun/09/2022)
   - Our paper was marked as state of the art in [Papers with Code](https://paperswithcode.com/task/real-time-semantic-segmentation). (Jun/06/2022)
   - Our paper was submitted to arXiv for public access. (Jun/04/2022)
   - The training and testing codes and trained models for PIDNet are available here. (Jun/03/2022)

## Demos

A demo of the segmentation performance of our proposed PIDNets: Original video (left) and predictions of PIDNet-S (middle) and PIDNet-L (right)
<p align="center">
  <img src="figs/video1_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #1</span>
</p>

<p align="center">
  <img src="figs/video2_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #2</span>
</p>

## Overview
<p align="center">
  <img src="figs/pidnet.jpg" alt="overview-of-our-method" width="800"/></br>
  <span align="center">An overview of the basic architecture of our proposed Proportional-Integral-Derivative Network (PIDNet). </span> 
</p>
P, I and D branches are responsiable for detail preservation, context embedding and boundary detection, respectively.

### Detailed Implementation
<p align="center">
  <img src="figs/pidnet_table.jpg" alt="overview-of-our-method" width="500"/></br>
  <span align="center">Instantiation of the PIDNet for semantic segmentation. </span> 
</p>
For operation, "OP, N, C" means operation OP with stride of N and the No. output channel is C; Output: output size given input size of 1024; mxRB: m residual basic blocks; 2xRBB: 2 residual bottleneck blocks; OP<sub>1</sub>\OP<sub>2</sub>: OP<sub>1</sub> is used for PIDNet-L while OP<sub>1</sub> is applied in PIDNet-M and PIDNet-S. (m,n,C) are scheduled to be (2,3,32), (2,3,64) and (3,4,64) for PIDNet-S, PIDNet-M and PIDNet-L, respectively.

## Models
For simple reproduction, we provide the ImageNet pretrained models here.

| Model (ImageNet) | PIDNet-S | PIDNet-M | PIDNet-L |
|:-:|:-:|:-:|:-:|
| Link | [download](https://drive.google.com/file/d/1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-/view?usp=sharing) | [download](https://drive.google.com/file/d/1gB9RxYVbdwi9eO5lbT073q-vRoncpYT1/view?usp=sharing) | [download](https://drive.google.com/file/d/1Eg6BwEsnu3AkKLO8lrKsoZ8AOEb2KZHY/view?usp=sharing) |

Also, the finetuned models on Cityscapes and Camvid are available for direct application in road scene parsing.

| Model (Cityscapes) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| PIDNet-S | [78.8](https://drive.google.com/file/d/1JakgBam_GrzyUMp-NbEVVBPEIXLSCssH/view?usp=sharing) | [78.6](https://drive.google.com/file/d/1VcF3NXLQvz2qE3LXttpxWQSdxTbATslO/view?usp=sharing) | 93.2 |
| PIDNet-M | [79.9](https://drive.google.com/file/d/1q0i4fVWmO7tpBKq_eOyIXe-mRf_hIS7q/view?usp=sharing) | [79.8](https://drive.google.com/file/d/1wxdFBzMmkF5XDGc_LkvCOFJ-lAdb8trT/view?usp=sharing) | 42.2 |
| PIDNet-L | [80.9](https://drive.google.com/file/d/1AR8LHC3613EKwG23JdApfTGsyOAcH0_L/view?usp=sharing) | [80.6](https://drive.google.com/file/d/1Ftij_vhcd62WEBqGdamZUcklBcdtB1f3/view?usp=sharing) | 31.1 |

| Model (CamVid) | Val (% mIOU) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|:-:|
| PIDNet-S |-| [80.1](https://drive.google.com/file/d/1h3IaUpssCnTWHiPEUkv-VgFmj86FkY3J/view?usp=sharing) | 153.7 |
| PIDNet-M |-| [82.0](https://drive.google.com/file/d/1rNGTc8LD42h8G3HaedtqwS0un4_-gEbB/view?usp=sharing) | 85.6 |

## Prerequisites
This implementation is based on [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation). Please refer to their repository for installation and dataset preparation. The inference speed is tested on single RTX 3090 using the method introduced by [SwiftNet](https://arxiv.org/pdf/1903.08469.pdf). No third-party acceleration lib is used, so you can try [TensorRT](https://github.com/NVIDIA/TensorRT) or other approaches for faster speed.

## Usage

### 0. Prepare the dataset

* Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets and unzip them in `data/cityscapes` and `data/camvid` dirs.
* Check if the paths contained in lists of `data/list` are correct for dataset images.

#### :smiley_cat: Instruction for preparation of CamVid data (remains discussion) :smiley_cat:

* Download the images and annotations from [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid), where the resolution of images is 960x720 (original);
* Unzip the data and put all the images and all the colored labels into `data/camvid/images/` and `data/camvid/labels`, respectively;
* Following the split of train, val and test sets used in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial), we have generated the dataset lists in `data/list/camvid/`;
* Finished!!! (We have open an issue for everyone who's interested in CamVid to discuss where to download the data and if the split in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial) is correct. BTW, do not directly use the split in [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid), which is wrong and will lead to unnormal high accuracy. We have revised the CamVid content in the paper and you will see the correct results after its announcement.)
### 1. Training

* Download the ImageNet pretrained models and put them into `pretrained_models/imagenet/` dir.
* For example, train the PIDNet-S on Cityscapes with batch size of 12 on 2 GPUs:
````bash
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml GPUS (0,1) TRAIN.BATCH_SIZE_PER_GPU 6
````
* Or train the PIDNet-L on Cityscapes using train and val sets simultaneously with batch size of 12 on 4 GPUs:
````bash
python tools/train.py --cfg configs/cityscapes/pidnet_large_cityscapes_trainval.yaml GPUS (0,1,2,3) TRAIN.BATCH_SIZE_PER_GPU 3
````

### 2. Evaluation

* Download the finetuned models for Cityscapes and CamVid and put them into `pretrained_models/cityscapes/` and `pretrained_models/camvid/` dirs, respectively.
* For example, evaluate the PIDNet-S on Cityscapes val set:
````bash
python tools/eval.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt
````
* Or, evaluate the PIDNet-M on CamVid test set:
````bash
python tools/eval.py --cfg configs/camvid/pidnet_medium_camvid.yaml \
                          TEST.MODEL_FILE pretrained_models/camvid/PIDNet_M_Camvid_Test.pt \
                          DATASET.TEST_SET list/camvid/test.lst
````
* Generate the testing results of PIDNet-L on Cityscapes test set:
````bash
python tools/eval.py --cfg configs/cityscapes/pidnet_large_cityscapes_trainval.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt \
                          DATASET.TEST_SET list/cityscapes/test.lst
````

### 3. Speed Measurement

* Measure the inference speed of PIDNet-S for Cityscapes:
````bash
python models/speed/pidnet_speed.py --a 'pidnet-s' --c 19 --r 1024 2048
````
* Measure the inference speed of PIDNet-M for CamVid:
````bash
python models/speed/pidnet_speed.py --a 'pidnet-m' --c 11 --r 720 960
````

### 4. Custom Inputs

* Put all your images in `samples/` and then run the command below using Cityscapes pretrained PIDNet-L for image format of .png:
````bash
python tools/custom.py --a 'pidnet-l' --p '../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt' --t '.png'
````

## Citation

If you think this implementation is useful for your work, please cite our paper:
```
@misc{xu2022pidnet,
      title={PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller}, 
      author={Jiacong Xu and Zixiang Xiong and Shankar P. Bhattacharyya},
      year={2022},
      eprint={2206.02066},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

* Our implementation is modified based on [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation).
* Latency measurement code is borrowed from the [DDRNet](https://github.com/ydhongHIT/DDRNet).
* Thanks for their nice contribution.

