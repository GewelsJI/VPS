# <p align=center>`Deep Video Polyp Segmentation (VPS)`</p><!-- omit in toc -->

<p align="center">
    <img src="./assets/background-min.gif"/> <br />
</p>

- **Title:** Deep Learning for Video Polyp Segmentation: A Comprehensive Study ([arXiv]())
- **Authors:** [Ge-Peng Ji](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=oaxKYKUAAAAJ)^, Guobao Xiao^, [Yu-Cheng Chou](https://scholar.google.com/citations?user=YVNRBTcAAAAJ&hl=en)^, [Deng-Ping Fan](https://dengpingfan.github.io/)*, [Kai Zhao](https://kaizhao.net/), [Geng Chen](https://scholar.google.com/citations?user=sJGCnjsAAAAJ&hl=en), [Huazhu Fu](https://hzfu.github.io/), and [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en).
- **Contact:** Our ability is limited, so we are welcome to receive your suggestions and contributions to our project. Or if you have any questions about our project, please feel free to drop us an e-mail (gepengai.ji@gmail.com, johnson111788@gmail.com, dengpfan@gmail.com) or directly report it in the issue or push a PR. Your star is our motivation, thank you first, let's enjoy it!

# Contents<!-- omit in toc -->
- [1. Features](#1-features)
- [2. News](#2-news)
- [3. VPS Dataset](#3-vps-dataset)
- [4. VPS Baseline](#4-vps-baseline)
- [5. VPS Benchmark](#5-vps-benchmark)
- [6. Tracking Trends](#6-tracking-trends)
- [7. Citations](#7-citations)
- [8. License](#8-license)
- [9. Acknowledgements](#9-acknowledgements)


# 1. Features

In the deep learning era, we present the first comprehensive video polyp segmentation (VPS) study. Over the years, developments on VPS are not moving forward with ease since large-scale fine-grain segmentation masks are still not made publicly available. To tackle
this issue, we first introduce a high-quality per-frame annotated VPS dataset. There are four features of our work:

- **VPS Dataset:** We recognize the importance of annotated medical data for substantial progress in research on medical AI systems’ development. And thus, our SUN-SEG dataset is open access, a non-profit database of the high-quality, large-scale, densely-annotated dataset for facilitating the colonoscopy diagnosis, localization, and derivative tasks. Our vision aims to provide data and knowledge to aid and educate clinicians, and also for the development of automated medical decision support systems.
- **VPS Baseline:** We propose a simple but strong baseline, which outperforms the cutting-edge polyp segmentation approaches and run in super real-time (170fps). We hope such a baseline could attract more researchers to join our community and inspire them to develop more interesting solutions.
- **VPS Benchmark:** For a fair comparison, we build an online leaderboard to keep up with the new progress of other competitors. Besides, we provide an out-of-the-box evaluation toolbox for the VPS task.
- **Tracking Trends:** We elaborately collect a paper reading list to continuously track the latest updates in this rapidly advancing field.


# 2. News

- *[March/16/2022]* Upload the training/testing code for our enhanced model PNS+.
- *[March/15/2022]* Release the evaluation toolbox for the VPS task. Add the [AWESOME_VPS.md] for tracking this field.
- *[March/14/2022]* Create the project page.


# 3. VPS Dataset

<p align="center">
    <img src="./assets/Pathological-min.gif"/> <br />
    <em> 
    Figure 2: 
    </em>
</p>

Notably, based on some necessary privacy-preserving considerations from the SUN dataset, we could not directly share the download link of the video dataset with you without authorization. And please inform us of your institution and the purpose of using SUN-SEG in the email. Thank you for your understanding! 

- How do get access to our SUN-SEG dataset? Please refer to [`DATA_PREPARATION`](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md).
- If you wanna know more descriptions about our SUN-SEG dataset. Please refer to our [`DATA_DESCRIPTION.md`](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_DESCRIPTION.md).


# 4. VPS Baseline


This work is the extension version of our conference paper (Progressively Normalized Self-Attention Network for Video Polyp Segmentation) accepted at MICCAI-2021. More details could refer to [arXiv](https://arxiv.org/abs/2105.08468) and [Github Link](https://github.com/GewelsJI/PNS-Net)


# 5. VPS Benchmark

[insert a benchmark table]

We provide an out-of-the-box evaluation toolbox for the VPS task, which is written in python style. You can just run it to generate the evaluation results on your custom approach. Or you can directly download the complete VPS benchmark toolbox (including ground-truth, prediction map of each competitor, and evaluation toolbox code) at [`DownloadLink`](). 

- More instructions about **Evaluation Toolbox** refer to [`PageLink`](https://github.com/GewelsJI/VPS/tree/main/eval).

We also build an online leaderboard to keep up with the new progress of other competitors. We believe this is a fun way to learn about new research directions and stay in tune with our VPS community.

- Online leaderboard is publicly avaliable at [`PaperWithCode`](). 

# 6. Tracking Trends

<p align="center">
    <img src="./assets/the-reading-list.png"/> <br />
</p>

In order to better understand the development of this field and to help researchers in their research process, we build the comprehensive **Paper Reading List**. It includes **119** colonoscopy imaging-based AI scientific research in recent **12** years. It includes several fields, such as image polyp segmentation, video polyp segmentation, image polyp detection, video polyp detection, and image polyp classification. We also list some interesting resources about human colonoscopy.

- The latest paper reading list and some interesting resources refer to [`Awesome-Video-Polyp-Segmentation.md`](https://github.com/GewelsJI/VPS/blob/main/docs/AWESOME_VPS.md)

# 7. Citations

If you have found our work useful, please use the following reference to cite this project:

    @article{ji2022vps,
        title={Deep Learning for Video Polyp Segmentation: A Comprehensive Study},
        author={Ji, Ge-Peng and Xiao, Guobao and Chou, Yu-Cheng and Fan, Deng-Ping and Zhao, Kai and Chen, Geng and Fu, Huazhu and Gool, Luc Van},
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

# 8. License

The dataset and source code is free for research and education use only. Any commercial usage should get formal permission first.

- **Video Source:** SUN (Showa University and Nagoya University) Colonoscopy Video Database is the colonoscopy-video database for the evaluation of automated colorectal-polyp detection. The database comprises still images of videos, which are collected at the Showa University Northern Yokohama Hospital. Mori Laboratory, Graduate School of Informatics, Nagoya University developed this database. Every frame in the database was annotated by the expert endoscopists at Showa University.

- **Intended Use:** This database is available for only non-commercial use in research or educational purpose. 
As long as you use the database for these purposes, you can edit or process images and annotations in this database. 
Without permission from Mori Lab., commercial use of this dataset is prohibited even after copying, editing, 
processing or any operations of this database. Please contact us for commercial use or if you are uncertain about
the decision.

- **Distribution:** It is prohibited to sell, transfer, lend, lease, resell, distribute, etc., as it is, or copy, edit, or process this database, in whole or in part.



# 9. Acknowledgements

Our dataset is built upon SUN (Showa University and Nagoya University) Colonoscopy Video Database, thanks very much for their wonderful work!