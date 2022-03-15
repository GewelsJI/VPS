# Deep Video Polyp Segmentation (VPS)

Official project page of our manuscript titled '*Deep Learning for Video Polyp Segmentation: A Comprehensive Study*'. [Paper Link]()



<img src="https://drive.google.com//uc?export=view&id=14FfYD9pHEDEoh4qnP0EYjDQZINx3mn0w" style="zoom:200%;" />

> **Authors:** [Ge-Peng Ji](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=oaxKYKUAAAAJ)^, Guobao Xiao^, [Yu-Cheng Chou](https://scholar.google.com/citations?user=YVNRBTcAAAAJ&hl=en)^, [Deng-Ping Fan](https://dengpingfan.github.io/)*, [Kai Zhao](https://kaizhao.net/), [Geng Chen](https://scholar.google.com/citations?user=sJGCnjsAAAAJ&hl=en), [Huazhu Fu](https://hzfu.github.io/), and [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en).

- **Contact:** We believe that the power of one man is limited, so we are welcome to receive your suggestions and contributions to our project. If you have any questions, please feel free contact us via e-mail (gepengai.ji@gmail.com) or directly new a issue on the GitHub.


# Features

- **VPS Dataset:** We recognize the importance of annotated medical data for substantial progress in research on medical AI systems’ development. And thus, SUN-SEG is an open access, non-profit database of high-quality, large-scale, densely-annotated dataset for faciliating the colonoscopy diagnosis, localization, and derivative tasks. Our vision aims to provide data and knowledge to aid and educate clinicians, and also for the development of automated medical decision support systems. Please refer to [PageLink]().
- **VPS Baseline:** We propose a simple but strong baseline, which outperforms the cutting-edge polyp segmentation approaches and run in super real-time (170fps). We hope such baseline could attract more researchers to join our community and inspire them to develop more interesting solutions. Please refer to [PageLink]().
- **VPS Benchmark:** For a fair comparison, we build an online leaderboard to keep up with new progress of other competitors. Please refer to [PageLink](). We also provide a evaluation toolbox for VPS task.
- **Tracking Trends:** We elaborately collect a paper reading list to continuously track the latest updates in this rapidly advancing field. Please refer to [PageLink]().

# :fire:News:fire:

- *[March/17/2022]* Upload the training/testing code for our enhanced model PNS+.
- *[March/16/2022]* 
- *[March/15/2022]* Release the evaluation toolbox for VPS task. Add the [AWESOME_VPS.md] for tracking this field.
- *[March/14/2022]* Initial the project.


# VPS Dataset

![](https://drive.google.com//uc?export=view&id=1RU6kIRn3ZcZiI1sw4WA19mBQkqAt851U)

Based on some necessary privacy-perserving considerations, we could not directly share the download link of ground-truth with you. And please inform us of your institution and the purpose of using SUN-SEG in the email. If you have any questions, please feel free to contact us by e-mail (gepengai.ji@gmail.com). Thank you for your understanding! 

- More details about **Dataset Preparation** refer to [PageLink]().
- More details about **Dataset Description** refer to [PageLink]().


# VPS Baseline

## 

# VPS Benchamrk

We provide a out-of-the-box evaluation toolbox for VPS task, which is written in python style. You can just run it to generate the evaluation results on your custom apporach. Or you can directly download the complete VPS benchmark toolbox (including ground-truth, prediction map of each competitor, and evaluation toolbox code) at [PageLink](). 

- More instructions about **Evaluation Toolbox** refer to [PageLink]().
- Online leaderboard is public at PaperWithCode. We believe this is a fun way to learn about new research directions and staying in tune with our VPS community.

# Citations

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

# Acknowledgements

# LICENSE

## Original statement

- **Video Source:** SUN (Showa University and Nagoya University) Colonoscopy Video Database is the colonoscopy-video database for the evaluation of an automated colorectal-polyp detection. The database comprises of still images of videos, which are collected at the Showa University Northern Yokohama Hospital. Mori Laboratory, Graduate School of Informatics, Nagoya University developed this database. Every frame in the database was annotated by the expert endoscopists at Showa University.

- **Intended Use:** This database is available for only non-commercial use in research or educational purpose. 
As long as you use the database for these purposes, you can edit or process images and annotations in this database. 
Without permission from Mori Lab., commercial use of this dataset is prohibited even after copying, editing, 
processing or any operations of this database. Please contact us for commercial use or if you are uncertain about
the decision.

- **Distribution:** It is prohibited to sell, transfer, lend, lease, resell, distribute, etc., as it is, or copy, edit, or process this database, in whole or in part.