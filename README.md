# <p align=center>`Video Polyp Segmentation: A Deep Learning Perspective`</p><!-- omit in toc -->

[![license: mit](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![LAST COMMIT](https://img.shields.io/github/last-commit/GewelsJI/VPS?style=flat-square)
![ISSUES](https://img.shields.io/github/issues/GewelsJI/VPS?style=flat-square)
![STARS](https://img.shields.io/github/stars/GewelsJI/VPS?style=flat-square)
[![ARXIV PAPER](https://img.shields.io/badge/Arxiv-Paper-red?style=flat-square)](https://arxiv.org/pdf/2203.14291.pdf)
[![Gitter](https://badges.gitter.im/video-polyp-segmentation/community.svg)](https://gitter.im/video-polyp-segmentation/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-polyp-segmentation-a-deep-learning/video-polyp-segmentation-on-sun-seg-easy)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-easy?p=video-polyp-segmentation-a-deep-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-polyp-segmentation-a-deep-learning/video-polyp-segmentation-on-sun-seg-hard)](https://paperswithcode.com/sota/video-polyp-segmentation-on-sun-seg-hard?p=video-polyp-segmentation-a-deep-learning)

<p align="center">
    <img src="./assets/background-min.gif"/> <br />
</p>

- **Title:** Video Polyp Segmentation: A Deep Learning Perspective ( [arXiv](https://arxiv.org/abs/2203.14291) )
- **Authors:** [Ge-Peng Ji](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=oaxKYKUAAAAJ)^, Guobao Xiao^, [Yu-Cheng Chou](https://scholar.google.com/citations?user=YVNRBTcAAAAJ&hl=en)^, [Deng-Ping Fan](https://dengpingfan.github.io/)*, [Kai Zhao](https://kaizhao.net/), [Geng Chen](https://scholar.google.com/citations?user=sJGCnjsAAAAJ&hl=en), [Huazhu Fu](https://hzfu.github.io/), and [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en).
- **Contact:** This project is still work in progress, and we invite all to contribute in making it more acessible and useful. If you have any questions, please feel free to drop us an e-mail (gepengai.ji@gmail.com, johnson111788@gmail.com, dengpfan@gmail.com) or directly report it in the issue or push a PR. Your star is our motivation, let's enjoy it!
- Welcome any discussions on video polyp segmentation at [Gitter room](https://gitter.im/video-polyp-segmentation/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link) or join our [WeChat group](https://github.com/GewelsJI/VPS/blob/main/assets/wechat_group.JPG).

# Contents<!-- omit in toc -->
- [1. Features](#1-features)
- [2. News](#2-news)
- [3. VPS Dataset](#3-vps-dataset)
- [4. VPS Baseline](#4-vps-baseline)
- [5. VPS Benchmark](#5-vps-benchmark)
- [6. Tracking Trends](#6-tracking-trends)
- [7. Citations](#7-citations)
- [8. FAQ](#8-faq)
- [9. License](#9-license)
- [10. Acknowledgements](#10-acknowledgements)


# 1. Features

In the deep learning era, we present the first comprehensive video polyp segmentation (VPS) study. Over the years, developments on VPS are not moving forward with ease since large-scale fine-grained segmentation masks are still not made publicly available. To tackle this issue, we first introduce a long-awaited high-quality per-frame annotated VPS dataset. There are four features of our work:

- **VPS Dataset:** We recognize the importance of annotated medical data for substantial progress in research on medical AI systems’ development. And thus, our SUN-SEG dataset is open access, a non-profit database of the high-quality, large-scale, densely-annotated dataset for facilitating the colonoscopy diagnosis, localization, and derivative tasks. Our vision aims to provide data and knowledge to aid and educate clinicians, and also for the development of automated medical decision support systems.
- **VPS Baseline:** We propose a simple but efficient baseline, which outperforms the 13 cutting-edge polyp segmentation approaches and run in super real-time (170fps). We hope such a baseline could attract more researchers to join our community and inspire them to develop more interesting solutions.
- **VPS Benchmark:** For a fair comparison, we build an online leaderboard to keep up with the new progress of VPS community. Besides, we provide an out-of-the-box evaluation toolbox for the VPS task.
- **Tracking Trends:** We elaborately collect a paper reading list to continuously track the latest updates in this rapidly advancing field.


# 2. News
 
- *[May/11/2022]* Release rejected labels: [SUN-SEG-Rejected-Labels](https://drive.google.com/file/d/1OtK2PR6gKQv56dIFjw0rXadcgGonf93S/view?usp=sharing). More details see [here](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_DESCRIPTION.md#rejected-labels).
- *[April/04/2022]* Create PaperWithCode page: [SUN-SEG-Easy](https://paperswithcode.com/dataset/sun-seg-easy) & [SUN-SEG-Hard](https://paperswithcode.com/dataset/sun-seg-hard).
- *[March/27/2022]* Release pretrained checkpoints and whole benchamrks results.
- *[March/18/2022]* Upload the whole training/testing code for our enhanced model PNS+.
- *[March/15/2022]* Release the evaluation toolbox for the VPS task. Add a [Awesome_Video_Polyp_Segmentation.md](https://github.com/GewelsJI/VPS/blob/main/docs/AWESOME_VPS.md) for tracking latest trends of this community.
- *[March/14/2022]* Create the project page.


# 3. VPS Dataset

<p align="center">
    <img src="./assets/Pathological-min.gif"/> <br />
    <em> 
    Figure 1: Annotation of SUN-SEG dataset. The object-level segmentation masks in SUN-SEG dataset of different pathological categories, which is densely annotated with experienced annotators and verified by colonoscopy-related researchers to ensure the quality of the proposed dataset. 
    </em>
</p>

Notably, based on some necessary privacy-preserving considerations from the SUN dataset, we could not directly share the download link of the video dataset with you without authorization. And please inform us of your institution and the purpose of using SUN-SEG in the email. Thank you for your understanding! 

- How to get access to our SUN-SEG dataset? Please refer to [`DATA_PREPARATION`](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md).
- If you wanna know more descriptions about our SUN-SEG dataset. Please refer to our [`DATA_DESCRIPTION.md`](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_DESCRIPTION.md).


# 4. VPS Baseline

> This work is the extension version of our conference paper (Progressively Normalized Self-Attention Network for Video Polyp Segmentation) accepted at MICCAI-2021. More details could refer to [arXiv](https://arxiv.org/abs/2105.08468) and [Github Link](https://github.com/GewelsJI/PNS-Net)


<p align="center">
    <img src="./assets/PNSPlus-Framework.png"/> <br />
    <em> 
    Figure 2: The pipeline of the proposed (a) PNS+ network, which is based on (b) the normalized self-attention (NS) block.
    </em>
</p>



There are three simple-to-use steps to access our project code (PNS+):

- Prerequisites of environment: 

  ```bash
    conda create -n PNS+ python=3.6
    conda activate PNS+
    conda install pytorch=1.1.0 torchvision -c pytorch
    pip install tensorboardX tqdm Pillow==6.2.2
    pip install git+https://github.com/pytorch/tnt.git@master
    ```

- Compiling the project:
    ```bash
    cd ./lib/PNS
    python setup.py build develop
    ```
- Training:

    ```bash
    python ./scripts/my_train.py
    ```

- Testing:
    
    Downloading pre-trained weights and move it into `snapshot/PNSPlus/epoch_15/PNSPlus.pth`, 
    which can be found in this download link: [Google Drive](https://drive.google.com/drive/folders/1NboArbmI4qg9AMQA0VoJmg4g1FaMUh4Q?usp=sharing) / [Baidu Drive](https://pan.baidu.com/s/1mnd9GD2BiWFzsibv7WiwAA) (Password: g7sa, Size: 108MB).
    ```bash
    python ./scripts/my_test.py
    ```


# 5. VPS Benchmark

We provide an out-of-the-box evaluation toolbox for the VPS task, which is written in Python style. You can just run it to generate the evaluation results on your custom approach. Or you can directly download the complete VPS benchmark including prediction map of each competitor at download link: [Google Drive](https://drive.google.com/drive/folders/15xBWCFuWWg9YL5p4qdcfz8-yEeJlNFxg?usp=sharing) / [Baidu Drive](https://pan.baidu.com/s/1plDtddL9L_f0dqrREVkIPg) (Password: 9c4p, Size: 7.28G).

- More instructions about **Evaluation Toolbox** refer to [`PageLink`](https://github.com/GewelsJI/VPS/tree/main/eval).

We also build an online leaderboard to keep up with the new progress of other competitors. We believe this is a fun way to learn about new research directions and stay in tune with our VPS community.

- Online leaderboard is publicly avaliable at PaperWithCode: [SUN-SEG-Easy](https://paperswithcode.com/dataset/sun-seg-easy) & [SUN-SEG-Hard](https://paperswithcode.com/dataset/sun-seg-hard).

Here, we present a variety of qualitative and quantitative results of VPS benchamrk:


- Visual prediction of top-performance competitors:

<p align="center">
    <img src="./assets/Qual-min.gif"/> <br />
    <em> 
    Figure 3: Qualitative comparison of three video-based models (PNS+, PNSNet, and 2/3D) and two image-based models (ACSNet, and PraNet).  
    </em>
</p>

- Model-based performance:

<p align="center">
    <img src="./assets/ModelPerformance.png"/> <br />
    <em> 
    Figure 4: Quantitative comparison on two testing sub-datasets, i.e., SUN-SEG-Easy and SUN-SEG-Hard. `R/T' represents we re-train the non-public model, whose code is provided by the original authors. The best scores are highlighted in bold.
    </em>
</p>

- Attribute-based performance:

<p align="center">
    <img src="./assets/AttributePerformance.png"/> <br />
    <em> 
    Figure 5: Visual attributes-based performance on our SUN-SEG-Easy and SUN-SEG-Hard in terms of structure measure.
    </em>
</p>


# 6. Tracking Trends

<p align="center">
    <img src="./assets/the-reading-list.png"/> <br />
</p>

To better understand the development of this field and to quickly push researchers in their research process, we elaborately build a **Paper Reading List**. It includes **119** colonoscopy imaging-based AI scientific research in recent 12 years. It includes several fields, such as image polyp segmentation, video polyp segmentation, image polyp detection, video polyp detection, and image polyp classification. Besides, we will provide some interesting resources about human colonoscopy. 

> **Note:** If we miss some treasure works, please let me know via e-mail or directly push a PR. We will work on it as soon as possible. Many thanks for your active feedbacks.

- The latest paper reading list and some interesting resources refer to [`Awesome-Video-Polyp-Segmentation.md`](https://github.com/GewelsJI/VPS/blob/main/docs/AWESOME_VPS.md)


# 7. Citations

If you have found our work useful, please use the following reference to cite this project:

    @article{ji2022vps,
        title={Deep Learning for Video Polyp Segmentation: A Comprehensive Study},
        author={Ji, Ge-Peng and Xiao, Guobao and Chou, Yu-Cheng and Fan, Deng-Ping and Zhao, Kai and Chen, Geng and Fu, Huazhu and Van Gool, Luc},
        journal={arXiv},
        year={2022}
    }

    @inproceedings{ji2021pnsnet,
        title={Progressively Normalized Self-Attention Network for Video Polyp Segmentation},
        author={Ji, Ge-Peng and Chou, Yu-Cheng and Fan, Deng-Ping and Chen, Geng and Jha, Debesh and Fu, Huazhu and Shao, Ling},
        booktitle={MICCAI},
        pages={142--152},
        year={2021}
    }

# 8. FAQ

- Thanks to [Tuo Wang (王拓)](victor_wt@qq.com) for providing a great solution to [upgrade the CUDA version when compling the NS block](./docs/Upgrade%20environment%20for%20NS%20block.pdf).

# 9. License

The dataset and source code is free for research and education use only. Any commercial usage should get formal permission first.

- **Video Source:** SUN (Showa University and Nagoya University) Colonoscopy Video Database is the colonoscopy-video database for the evaluation of automated colorectal-polyp detection. The database comprises still images of videos, which are collected at the Showa University Northern Yokohama Hospital. Mori Laboratory, Graduate School of Informatics, Nagoya University developed this database. Every frame in the database was annotated by the expert endoscopists at Showa University.

- **Intended Use:** This database is available for only non-commercial use in research or educational purpose. 
As long as you use the database for these purposes, you can edit or process images and annotations in this database. 
Without permission from Mori Lab., commercial use of this dataset is prohibited even after copying, editing, 
processing or any operations of this database. Please contact us for commercial use or if you are uncertain about
the decision.

- **Distribution:** It is prohibited to sell, transfer, lend, lease, resell, distribute, etc., as it is, or copy, edit, or process this database, in whole or in part.


# 10. Acknowledgements

- Our dataset is built upon SUN (Showa University and Nagoya University) Colonoscopy Video Database, thanks very much for their wonderful work!
- This codebase is based on our conference version [PNSNet](https://github.com/GewelsJI/PNS-Net), which is accepted by MICCAI-2021 conference.
