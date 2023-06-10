# Awesome Video Polyp Segmentation

<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> 
<img src="https://img.shields.io/badge/Number%20of%20Papers-192-FF6F00" alt="PaperNum"/>

![](../assets/the-reading-list.png)

# 1. Preview

This is a paper collection of **133** colonoscopy imaging-based AI scientific researches in recent **12** years.

In order to better understand the development of this field and to help researchers in their research process, we have divided the works into five tasks, including **133** papers on [image polyp segmentation](#21-image-polyp-segmentation), **5** papers on [video polyp segmentation](#22-video-polyp-segmentation), **17** papers on [image polyp detection](#23-image-polyp-detection),  **11** papers on [video polyp detection](#24-video-polyp-detection), **6** paper on [image polyp classification](#25-image-polyp-classification),  **1** paper on [video polyp classification](#26-video-polyp-classification), **2** paper on [colonoscopy depth estimation](#27-colonoscopy-depth-estimation), **1** paper on [colonoscopy deficient coverage detection](#28-colonoscopy-deficient-coverage-detection), and **1** paper on [colon polyp image synthesis](#29-colon-polyp-image-synthesis).

Besides, we present the collection of **14** polyp related datasets, including **8** [image segmentation datasets](#31-image-segmentation-datasets), **2** [video segmentation datasets](#32-video-segmentation-datasets), **1** [video detection dataset](#33-video-detection-datasets), **3** [video classification datasets](#34-video-classification-datasets), and **2** [colonoscopy depth dataset](#35-colonoscopy-depth-datasets).

In addition, we provide links to each paper and its repository whenever possible. * denotes the corresponding paper cannot be downloaded or the link is connected to the journal.

> Note that this page is under construction. If you have anything to recommend or any suggestions, please feel free to contact us via e-mail (gepengai.ji@gmail) or directly push a PR. 


--- *Last updated: 12/07/2022* --- 

## 1.1. Table of Contents

- [Awesome Video Polyp Segmentation](#awesome-video-polyp-segmentation)
- [1. Preview](#1-preview)
  * [1.1. Table of Contents](#11-table-of-contents)
- [2. Polyp Related Methods](#2-polyp-related-methods)
  * [2.1 Image Polyp Segmentation](#21-image-polyp-segmentation)
  * [2.2 Video Polyp Segmentation](#22-video-polyp-segmentation)
  * [2.3 Image Polyp Detection](#23-image-polyp-detection)
  * [2.4 Video Polyp Detection](#24-video-polyp-detection)
  * [2.5 Image Polyp Classification](#25-image-polyp-classification)
  * [2.6 Video Polyp Classification](#26-video-polyp-classification)
  * [2.7 Colonoscopy Depth Estimation](#27-colonoscopy-depth-estimation)
  * [2.8 Colonoscopy Deficient Coverage Detection](#28-colonoscopy-deficient-coverage-detection)
  * [2.9 Colon Polyp Image Synthesis](#29-colon-polyp-image-synthesis)
- [3. Polyp Related Datasets](#3-useful-resources)
  * [3.1 Image Segmentation Datasets](#31-image-segmentation-datasets)
  * [3.2 Video Segmentation Datasets](#32-video-segmentation-datasets)
  * [3.3 Video Detection Datasets](#33-video-detection-datasets)
  * [3.4 Video Classification Datasets](#34-video-classification-datasets)
  * [3.5 Colonoscopy Depth Datasets](#35-colonoscopy-depth-datasets)
- [4. Useful Resources](#4-useful-resources)
  * [4.1 Colonoscopy Related](#41-colonoscopy-related)
  * [4.2 AI Conference Deadlines](#42-ai-conference-deadlines)
  
# 2. Polyp Related Methods

## 2.1 Image Polyp Segmentation

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:
2023 | **PR** | Cross-level Feature Aggregation Network for Polyp Segmentation | [Paper](https://www.sciencedirect.com/science/article/pii/S0031320323002558)/[Code](https://github.com/taozh2017/CFANet)
2022 | **ISICT** | Incremental Boundary Refinement using Self Axial Reverse Attention and Uncertainty-aware Gate for Colon Polyp Segmentation | [Paper*](https://dl.acm.org/doi/abs/10.1145/3568562.3568663)/Code
2022 | **TVCJ** | DCANet: deep context attention network for automatic polyp segmentation | [Paper](https://link.springer.com/content/pdf/10.1007/s00371-022-02677-x.pdf?pdf=button)/Code
2022 | **arXiv** | Towards Automated Polyp Segmentation Using Weakly- and Semi-Supervised Learning and Deformable Transformers | [Paper](https://arxiv.org/pdf/2211.11847.pdf)/Code
2022 | **arXiv** | Spatially Exclusive Pasting: A General Data Augmentation for the Polyp Segmentation | [Paper](https://arxiv.org/pdf/2211.08284.pdf)/Code 
2022 | **CBM** | MSRAformer: Multiscale spatial reverse attention network for polyp segmentation | [Paper*](https://www.sciencedirect.com/science/article/abs/pii/S0010482522009829)/Code
2022 | **CBM** | DBMF: Dual Branch Multiscale Feature Fusion Network for polyp segmentation | [Paper*](https://www.sciencedirect.com/science/article/abs/pii/S0010482522010125)/Code
2022 | **ICAISM** | Automatic Polyp Segmentation in Colonoscopy Images Using Single Network Model: SegNet | [Paper](https://link.springer.com/content/pdf/10.1007/978-981-16-2183-3_69.pdf?pdf=inline%20link)/Code
2022 | **ICMAR** | Polyp segmentation algorithm combining multi-scale attention and multi-layer loss | [Paper*](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12331/123314W/Polyp-segmentation-algorithm-combining-multi-scale-attention-and-multi-layer/10.1117/12.2652907.short?SSO=1)/Code
2022 | **MICCAI** | Using Guided Self-Attention with Local Information for Polyp Segmentation | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_60)/Code
2022 | **MICCAI** | Task-Relevant Feature Replenishment for Cross-Centre Polyp Segmentation|  [Paper](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_57)/[Code](https://github.com/CathyS1996/TRFRNet)
2022 | **MICCAI** | TGANet: Text-guided attention for improved polyp segmentation | [Paper](https://arxiv.org/pdf/2205.04280.pdf)/[Code](https://github.com/nikhilroxtomar/TGANet)
2022 | **MICCAI** | Lesion-Aware Dynamic Kernel for Polyp Segmentation | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_10)/Code
2022 | **MICCAI** | Semi-Supervised Spatial Temporal Attention Network for Video Polyp Segmentation | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_7)/Code
2022 | **CMIG** | Boosting medical image segmentation via conditional-synergistic convolution and lesion decoupling | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0895611122000817)/[Code](https://github.com/QianChen98/CCLD-Net)
2022 | **Gastroenterology Insights** | UPolySeg: A U-Net-Based Polyp Segmentation Network Using Colonoscopy Images | [Paper](https://www.mdpi.com/2036-7422/13/3/27/pdf?version=1660117945)/Code
2022 | **Electronics** | A Segmentation Algorithm of Colonoscopy Images Based on Multi-Scale Feature Fusion | [Paper](https://www.mdpi.com/2079-9292/11/16/2501/pdf?version=1660199760)/Code
2022 | **TCSVT** | Polyp-Mixer: An Efficient Context-Aware MLP-based Paradigm for Polyp Segmentation | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9852486)/[Code](https://github.com/shijinghuihub/Polyp-Mixer)
2022 | **TETCI** | Adaptive Context Exploration Network for Polyp Segmentation in Colonoscopy Images | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9852746)/Code
2022 | **IJCAI** | ICGNet: Integration Context-Based Reverse-Contour Guidance Network for Polyp Segmentation | [Paper](https://www.ijcai.org/proceedings/2022/0123.pdf)/Code
2022 | **IJCAI** | TCCNet: Temporally Consistent Context-Free Network for Semi-supervised Video Polyp Segmentation | [Paper](https://www.ijcai.org/proceedings/2022/0155.pdf)/[Code](https://github.com/wener-yung/TCCNet)
2022 | **AIM** | An end-to-end tracking method for polyp detectors in colonoscopy videos | [Paper](https://reader.elsevier.com/reader/sd/pii/S0933365722001270?token=85406D788AF1B59597BD0BCA3456A70DD2ECE3040EBCB1C3D669584B712F58FA656224415F80070007005E3D36B81F8D&originRegion=us-east-1&originCreation=20220801152429)/Code
2022 | **MIUA** | Polyp2Seg: Improved Polyp Segmentation with Vision Transformer | [Paper](https://link.springer.com/content/pdf/10.1007/978-3-031-12053-4_39.pdf)/Code
2022 | **CVIP** | Localization of Polyps in WCE Images Using Deep Learning Segmentation Methods: A Comparative Study | [Paper](https://link.springer.com/content/pdf/10.1007/978-3-031-11346-8_46.pdf)/Code
2022 | **AIMD** | SARM-Net: A Spatial Attention-Based Residual M-Net for Polyp Segmentation | [Paper](https://link.springer.com/chapter/10.1007/978-981-19-0151-5_33)/Code
2022 | **BSPC** | FAPN: Feature Augmented Pyramid Network for polyp segmentation | [Paper](https://reader.elsevier.com/reader/sd/pii/S1746809422004074?token=FE011B0123F802F27442369ED87DD656B402B1920147F48DC6957C72D5D1859DFD0B1ADB80194C54FBB05A9220781C20&originRegion=us-east-1&originCreation=20220708075935)/Code
2022 | **BSPC** | Automated polyp segmentation in colonoscopy images via deep network with lesion-aware feature selection and refinement | [Paper](https://www.sciencedirect.com/sdfe/reader/pii/S1746809422003688/pdf)/Code
2022 | **ICFCS** | U-Shaped Xception-Residual Network for Polyps Region Segmentation | [Paper](https://link.springer.com/chapter/10.1007/978-981-19-0105-8_25)/Code
2022 | **CBM** | Colorectal polyp region extraction using saliency detection network with neutrosophic enhancement | [Paper](https://reader.elsevier.com/reader/sd/pii/S0010482522005340?token=0E8C163715B86CD6B6EC6F0D60685AA70877424A885FF787BB2C776923BB44A84AF65684FBCAC9503E28CA794B58174B&originRegion=us-east-1&originCreation=20220708080248)/Code
2022 | **IJCARS** | Examining the effect of synthetic data augmentation in polyp detection and segmentation | [Paper](https://link.springer.com/content/pdf/10.1007/s11548-022-02651-x.pdf)/Code
2022 | **IEEE Access** | Polyp Segmentation of Colonoscopy Images by Exploring the Uncertain Areas | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9775966)/Code
2022 | **JBHI** | Boundary Constraint Network with Cross Layer Feature Integration for Polyp Segmentation | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9772424)/Code
2022 | **CMIG** | Polyp Segmentation Network with Hybrid Channel-Spatial Attention and Pyramid Global Context Guided Feature Fusion | [Paper](https://reader.elsevier.com/reader/sd/pii/S0895611122000453?token=1CC9A6070522894EA16E7984593F634B2A32CCB11CB9CBFD4CAA37916A13DE4D3F6AB47DFDDB5F6ED12F23B0CAD0FE20&originRegion=us-east-1&originCreation=20220515094414)/Code
2022 | **arXiv** | Automatic Polyp Segmentation with Multiple Kernel Dilated Convolution Network | [Paper](https://arxiv.org/pdf/2206.06264.pdf)/Code
2022 | **arXiv** | PlutoNet: An Efficient Polyp Segmentation Network | [Paper](https://arxiv.org/pdf/2204.03652.pdf)/Code
2022 | **arXiv** | Automated Polyp Segmentation in Colonoscopy using MSRFNet | [Paper](https://www.researchgate.net/profile/Debesh-Jha/publication/359698512_Automated_Polyp_Segmentation_in_Colonoscopy_using_MSRFNet/links/624907bf8068956f3c6533c1/Automated-Polyp-Segmentation-in-Colonoscopy-using-MSRFNet.pdf)/Code
2022 | **arXiv** | BlazeNeo: Blazing fast polyp segmentation and neoplasm detection | [Paper](https://arxiv.org/pdf/2203.00129.pdf)/Code
2022 | **arXiv** | BDG-Net: Boundary Distribution Guided Network for Accurate Polyp Segmentation | [Paper](https://arxiv.org/pdf/2201.00767.pdf)/Code
2022 | **arXiv** | Cross-level Contrastive Learning and Consistency Constraint for Semi-supervised Medical Image Segmentation | [Paper](https://arxiv.org/pdf/2202.04074.pdf)/Code
2022 | **arXiv** | ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation | [Paper](https://arxiv.org/pdf/2205.08473.pdf)/Code
2022 | **KBS** | MIA-Net: Multi-information aggregation network combining transformers and convolutional feature learning for polyp segmentation | [Paper](https://www.sciencedirect.com/science/article/pii/S0950705122003926)/Code
2022 | **Diagnostics** | Performance of Convolutional Neural Networks for Polyp Localization on Public Colonoscopy Image Datasets | [Paper](https://www.mdpi.com/2075-4418/12/4/898/htm)/Code
2022 | **IEEE TCyber** | PolypSeg+: A Lightweight Context-Aware Network for Real-Time Polyp Segmentation | [Paper](https://ieeexplore.ieee.org/abstract/document/9756512)/Code
2022 | **JCDE** | SwinE-Net: hybrid deep learning approach to novel polyp segmentation using convolutional neural network and Swin Transformer | [Paper](https://academic.oup.com/jcde/article/9/2/616/6564811?login=true)/Code
2022 | **IEEE JBHI** | Artificial Intelligence for Colonoscopy: Past, Present, and Future | [Paper](https://ieeexplore.ieee.org/document/9739863)/Code
2021 | **ICONIP** | Multi-scale Fusion Attention Network for Polyp Segmentation | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-92310-5_19)/Code
2021 | **AAAI** | Precise yet Efficient Semantic Calibration and Refinement in ConvNets for Real-time Polyp Segmentation from Colonoscopy Videos | [Paper](https://www.aaai.org/AAAI21Papers/AAAI-5002.WuHS.pdf)/Code  
2021 | **ICCV** | Collaborative and Adversarial Learning of Focused and Dispersive Representations for Semi-supervised Polyp Segmentation | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_and_Adversarial_Learning_of_Focused_and_Dispersive_Representations_for_ICCV_2021_paper.pdf)/Code
2021 | **ACM MM** | UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation | [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475375)/[Code](https://github.com/plemeri/UACANet) 
2021 | **Healthcare** | TMD-Unet: Triple-Unet with Multi-Scale Input Features and Dense Skip Connection for Medical Image Segmentation | [Paper](https://www.researchgate.net/publication/348283572_TMD-Unet_Triple-Unet_with_Multi-Scale_Input_Features_and_Dense_Skip_Connection_for_Medical_Image_Segmentation)/Code 
2021 | **ICPR** | DDANet: Dual decoder attention network for automatic polyp segmentation | [Paper](https://arxiv.org/pdf/2012.15245.pdf)/[Code](https://github.com/nikhilroxtomar/DDANet)
2021 | **ICDSIT** | Sa-HarDNeSt: A Self-Attention Network for Polyp Segmentation | [Paper](https://dl.acm.org/doi/pdf/10.1145/3478905.3478942)/Code
2021 | **IJCAI** | Medical image segmentation using squeeze-and-expansion transformers | [Paper](https://arxiv.org/pdf/2105.09511.pdf)/[Code](https://github.com/askerlee/segtran) 
2021 | **IEEE ISBI** | DivergentNets: Medical Image Segmentation by Network Ensemble | [Paper](https://arxiv.org/pdf/2107.00283.pdf)/[Code](https://github.com/vlbthambawita/divergent-nets) 
2021 | **IEEE JBHI** | A comprehensive study on colorectal polyp segmentation with resunet++, conditional random field and test-time augmentation | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9314114)/[Code](https://github.com/DebeshJha/ResUNetPlusPlus-with-CRF-and-TTA)
2021 | **IEEE JBHI** | Mutual-prototype adaptation for cross-domain polyp segmentation | [Paper](https://sci-hub.se/downloads/2021-05-24/1a/yang2021.pdf?rand=61ab86a0abb28?download=true)/[Code](https://github.com/CityU-AIM-Group/MPA-DA) 
2021 | **MIA** | Dynamic-weighting hierarchical segmentation network for medical images | [Paper](https://pdf.sciencedirectassets.com/272154/1-s2.0-S1361841521X00059/1-s2.0-S1361841521002413/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDZ0ITNVi%2BvzuLOUKPQQajvIwWt9hsidNnvAiweckL5rgIgQv2G7DZIc6bMCiIaipFlvaKM8zJWKR%2BJ75Q6tnwBWVQq%2BgMIYBAEGgwwNTkwMDM1NDY4NjUiDMa65PEpU2yP8TutHirXA2%2FvwxG2Px%2B5huEgdCPa%2BWft55sBiAK71F3Ebz%2Fj7AJ2EHzmyFNE4%2BtBAHS%2Ft2dAE0l9X1a8DBEj2hj%2BnQfo5lfkj1bS6gxEny5IHEKozs9X%2BAt1l1rv7PIPXN6Eb6%2B5%2FQe24O%2Bu6iJyeiTbUS2Pk3kZCiyIbVNRygfDn6j8l5Ye1JqLSl8zljMiZJBZKWfE9pekhQbKPi5pyqflJmiBhZFxuI2YGjOhQ0LhY0fQKIqrAu0AWvFXBdv%2BX%2FxLHBStP911TVbrCl6zCf4m1Y%2FYmrRkEiR343YY0VuJ%2Bswg8p23Lf7nQ3tcSvKaJ5WmpINOPl3O%2BA7AdiqOkvF2%2BklOIfqpNSn1kA591KHwjYh4tSIQdlVCBYQIOFb5DGD3hfnwUwery0DdMRC0H4vgChsyEPGUiI2JVgr6WytCvKl3%2Bnm%2FlaxUopskmpT7S5QGGIYkKy56VoBL1yBP4SqlU1bWe%2FbujaO6mq6i5xjPP8W2l4OlCZIwKX79dq17AIN%2B1cZZYp84zVSjYpyn8qSXkxCnf3x8UNKws2TzALIMQl9YQKloABt%2BvLQYoN7P6MINo%2FcB%2FaC6HP3yXMuho7oorZtPgU5P3IHpaHSxod3iKD5uvh87P2h2noQt%2FTCQ9K2NBjqlAQj%2FqBYVx%2BDjQ%2BXuOFSdHTIw2tRNrxOxQsEtmxdxy7ej5ttMEQKnq3rhA1K6ETIoS5IH5RNNgGzijDwYiNGXy4tJd7k3tmoFnoqicAqyycG%2FPu2gzvYGKhVNh9bxllTdijLBfJRTMKSMFwyH1W3JqFZUDbT9qO7bwqciDPMerP6zF0KdGfjBrY%2FPWYJf%2FWOdhSBBTE62hfm%2FW573OuLkI1KwU1dWrw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211204T153800Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4BAUX3ZO%2F20211204%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=80ffde177a66c02f8f70db6588324bca239c48bf4366af906e2cf876c11dbcea&hash=854aafe64938f04f8ee0bf386f4c92d77219c66c67bad994e7d8806082bdafd1&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1361841521002413&tid=spdf-875a5c1f-84ac-43d9-aff2-8406cae898df&sid=ceb35d509987134d700962537e5a5b60cfe0gxrqa&type=client)/[Code](https://github.com/CityU-AIM-Group/DW-HieraSeg) 
2021 | **MICCAI** | Automatic Polyp Segmentation via Multi-scale Subtraction Network | [Paper](https://arxiv.org/pdf/2108.05082.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/MSNet)
2021 | **MICCAI** | CCBANet: Cascading Context and Balancing Attention for Polyp Segmentation | [Paper*](https://link.springer.com/content/pdf/10.1007%2F978-3-030-87193-2.pdf)/[Code](https://github.com/ntcongvn/CCBANet)
2021 | **MICCAI** | Constrained Contrastive Distribution Learning for Unsupervised Anomaly Detection and Localisation in Medical Images | [Paper](https://arxiv.org/pdf/2103.03423.pdf)/[Code](https://arxiv.org/pdf/2103.03423.pdf)
2021 | **MICCAI** | Double Encoder-Decoder Networks for Gastrointestinal Polyp Segmentation | [Paper](https://arxiv.org/pdf/2110.01939.pdf)/Code
2021 | **MICCAI** | HRENet: A Hard Region Enhancement Network for Polyp Segmentation | [Paper*](https://link.springer.com/content/pdf/10.1007%2F978-3-030-87193-2.pdf)/[Code](https://github.com/CathySH/HRENet)
2021 | **MICCAI** | Few-Shot Domain Adaptation with Polymorphic Transformers | [Paper](https://arxiv.org/pdf/2107.04805.pdf)/[Code](https://github.com/askerlee/segtran)
2021 | **MICCAI** | Learnable Oriented-Derivative Network for Polyp Segmentation | [Paper*](https://link.springer.com/content/pdf/10.1007%2F978-3-030-87193-2.pdf)/[Code](https://github.com/midsdsy/LOD-Net)
2021 | **MICCAI** | Shallow attention network for polyp segmentation | [Paper](https://arxiv.org/pdf/2108.00882.pdf)/[Code](https://github.com/weijun88/SANet) 
2021 | **MICCAI** | Transfuse: Fusing transformers and cnns for medical image segmentation | [Paper](https://arxiv.org/pdf/2102.08005.pdf)/Code 
2021 | **MIDL** | Deep ensembles based on stochastic activation selection for polyp segmentation | [Paper](https://arxiv.org/pdf/2104.00850.pdf)/[Code](https://github.com/LorisNanni/Deep-ensembles-based-on-Stochastic-Activation-Selection-for-Polyp-Segmentation) 
2021 | **NCBI** | MBFFNet: Multi-Branch Feature Fusion Network for Colonoscopy | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8317500/pdf/fbioe-09-696251.pdf)/Code
2021 | **RIVF** | AG-CUResNeSt: A novel method for colon polyp segmentation | [Paper](https://arxiv.org/pdf/2105.00402.pdf)/Code 
2021 | **Sensors** | A-DenseUNet: Adaptive Densely Connected UNet for Polyp Segmentation in Colonoscopy Images with Atrous Convolution | [Paper](https://www.mdpi.com/1424-8220/21/4/1441/pdf)/Code
2021 | **IEEE TIM** | Colon Polyp Detection and Segmentation Based on Improved MRCNN | [Paper](https://www.researchgate.net/publication/346985142_Colon_Polyp_Detection_and_Segmentation_based_on_improved_MRCNN)/Code 
2021 | **IEEE TIM** | Polyp-Net A Multimodel Fusion Network for Polyp Segmentation | [Paper](https://drive.google.com/file/d/1isi_Blz9ZAK4iPH5wKcEVw4-FSYuqNxm/view)/Code 
2021 | **IEEE TMI** | Graph-based Region and Boundary Aggregation for Biomedical Image Segmentation | [Paper](https://livrepository.liverpool.ac.uk/3140502/1/TMI_region_boundary2021.pdf)/[Code](https://github.com/smallmax00/Graph_Region_Boudnary)
2021 | **IEEE Access** | A Simple Generic Method for Effective Boundary Extraction in Medical Image Segmentation | [Paper*](https://ieeexplore.ieee.org/iel7/6287639/9312710/09495769.pdf)/Code
2021 | **IEEE Access** | CRF-EfficientUNet: An Improved UNet Framework for Polyp Segmentation in Colonoscopy Images With Combined Asymmetric Loss Function and CRF-RNN Layer | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9622208)/[Code](https://github.com/lethithuhong1302/CRF-EfficientUNet)
2021 | **IEEE Access** | Real-time polyp detection, localization and segmentation in colonoscopy using deep learning | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9369308)/[Code](https://github.com/DebeshJha/ColonSegNet) 
2021 | **IEEE Access** | Training on Polar Image Transformations Improves Biomedical Image Segmentation | [Paper](https://ieeexplore.ieee.org/iel7/6287639/9312710/09551998.pdf)/Code
2021 | **BioMed** | Automated Classification and Segmentation in Colorectal Images Based on Self-Paced Transfer Network | [Paper](https://www.hindawi.com/journals/bmri/2021/6683931/)/Code 
2021 | **CBM** | Focus U-Net: A novel dual attention-gated CNN for polyp segmentation during colonoscopy | [Paper](https://sci-hub.se/10.1016/j.compbiomed.2021.104815)/Code 
2021 | **CBMS** | Nanonet: Real-time polyp segmentation in video capsule endoscopy and colonoscopy | [Paper](https://arxiv.org/pdf/2104.11138.pdf)/[Code](https://github.com/DebeshJha/NanoNet) 
2021 | **CRV** | Enhanced u-net: A feature enhancement network for polyp segmentation | [Paper](https://arxiv.org/pdf/2105.00999.pdf)/[Code](https://github.com/rucv/Enhanced-U-Net) 
2021 | **IEEE DDCLS** | MSB-Net: Multi-Scale Boundary Net for Polyp Segmentation | [Paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9455514)/Code
2021 | **arXiv** | BI-GCN: Boundary-Aware Input-Dependent Graph Convolution Network for Biomedical Image Segmentation | [Paper](https://arxiv.org/pdf/2110.14775.pdf)/Code
2021 | **arXiv** | CaraNet: Context Axial Reverse Attention Network for Segmentation of Small Medical Objects | [Paper](https://arxiv.org/pdf/2108.07368.pdf)/Code
2021 | **arXiv** | DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation | [Paper](https://arxiv.org/pdf/2106.06716.pdf)/Code
2021 | **arXiv** | Duplex contextual relation network for polyp segmentation | [Paper](https://arxiv.org/pdf/2103.06725)/[Code](https://github.com/PRIS-CV/DCRNet) 
2021 | **arXiv** | Few-shot segmentation of medical images based on meta-learning with implicit gradients | [Paper](https://arxiv.org/pdf/2106.03223.pdf)/Code 
2021 | **arXiv** | GMSRF-Net: An improved generalizability with global multi-scale residual fusion network for polyp segmentation | [Paper](https://arxiv.org/pdf/2111.10614.pdf)/Code
2021 | **arXiv** | Hardnet-mseg: A simple encoder-decoder polyp segmentation neural network that achieves over 0.9 mean dice and 86 fps | [Paper](https://arxiv.org/pdf/2101.07172.pdf)/[Code](https://github.com/james128333/HarDNet-MSEG) 
2021 | **arXiv** | NeoUNet: Towards accurate colon polyp segmentation and neoplasm detection | [Paper](https://arxiv.org/pdf/2107.05023.pdf)/Code
2021 | **arXiv** | Polyp segmentation in colonoscopy images using u-net-mobilenetv2 | [Paper](https://arxiv.org/pdf/2103.15715.pdf)/Code 
2021 | **arXiv**  | Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers | [Paper](https://arxiv.org/pdf/2108.06932.pdf)/[Code](https://github.com/DengPingFan/Polyp-PVT)
2021 | **arXiv** | Self-supervised Multi-class Pre-training for Unsupervised Anomaly Detection and Segmentation in Medical Images | [Paper](https://arxiv.org/pdf/2109.01303.pdf)/Code
2020 | **IEEE Access** | Contour-Aware Polyp Segmentation in Colonoscopy Images Using Detailed Upsamling Encoder-Decoder Networks | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9096362)/Code 
2020 | **arXiv** | Automatic polyp segmentation using convolution neural networks | [Paper](https://arxiv.org/pdf/2004.10792)/Code 
2020 | **arXiv** | Boundary-aware Context Neural Network for Medical Image Segmentation | [Paper](https://arxiv.org/pdf/2005.00966v1.pdf)/Code 
2020 | **CBMS** | DoubleU-Net A Deep Convolutional Neural Network for Medical Image Segmentation | [Paper](https://arxiv.org/pdf/2006.04868)/[Code](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net) 
2020 | **HYDCON** | Polyps Segmentation using Fuzzy Thresholding in HSV Color Space | [Paper](https://sci-hub.se/downloads/2020-12-21/80/mandal2020.pdf?rand=61ab7ececa47d?download=true)/Code 
2020 | **ICARM** | Real-time Colonoscopy Image Segmentation Based on Ensemble Knowledge Distillation | [Paper](https://sci-hub.se/downloads/2020-11-09/c2/huang2020.pdf?rand=61ab7ea22a48f?download=true)/Code 
2020 | **IEEE ISBI** | SSN A stair-shape network for real-time polyp segmentation in colonoscopy images | [Paper](https://www.researchgate.net/profile/Ruiwei-Feng/publication/339782695_SSN_A_Stair-Shape_Network_for_Real-Time_Polyp_Segmentation_in_Colonoscopy_Images/links/5e6af8dd458515e555765049/SSN-A-Stair-Shape-Network-for-Real-Time-Polyp-Segmentation-in-Colonoscopy-Images.pdf)/Code 
2020 | **IEEE JBHI** | Multi-scale Context-guided Deep Network for Automated Lesion Segmentation with Endoscopy Images of Gastrointestinal Tract | [Paper](https://sci-hub.se/downloads/2020-06-18//db/wang2020.pdf?rand=61ab7f237a9b5?download=true)/Code 
2020 | **MIA** | Uncertainty and interpretability in convolutional neural networks for semantic segmentation of colorectal polyps | [Paper](https://pdf.sciencedirectassets.com/272154/1-s2.0-S1361841519X00092/1-s2.0-S1361841519301574/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFCBk%2FghOtJfQ62sxbB63ru7AOYC0IBJiHp%2F3HJrjaU%2BAiANHFD06HdAPYve6VbMJLdwkGojYzHjBDAL846ZP7ss%2FiqDBAiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAQaDDA1OTAwMzU0Njg2NSIMdDwgrrObNsP3CG8sKtcDt1QC6gBH9G9et33b26BRppTYQZooVUiDID3EvU6AAsmbNCFp5XTmqgUR5rK90BvBAFaKmWgIVzhR4Hzs2%2B1knkFJMtwbhLFSPjSZ6rx%2F4%2FNI2%2FyPeMY5xN3Qd1WdRfM4I2HF%2Bmy3aO2xX1j4IchIAPhE5FVuXDE1BqWefNjTLHxO9Or7i8FcFktna%2BIB1nwJVuXieWZNtyQhRwa%2BxRDo1TvyBZ3FXQ4UJ0wVQT6ndB6eBuKMXRJrDlBzg2MlKxIII65ufBi5GhVJLqY7lZCfFswH2DhBXPYvmnqi64jrhPq6i4iRMUu%2FU78b7ibopJgGeWbqg337XQppOTHxiGxll2dhvJ7PMb%2FtN7WTWIkEYGvPr%2FPCf9ccgo%2FJDJZi7n7u9h1%2BEqNU5YPgHmW6N%2FIK19qSQrUbKq35DMgajbzHtqOSUYww2g%2FeLn8Ymyty19TwfWC%2BWAJfT%2BgbJGt3c%2BhEudC2o%2B37%2BZ9dlAjFdGLNrV%2BEb2sTR6753%2FLynwAL2ateKKWQkTrurKfmPIdVnj0J7mLMx9QXZI0nrVzjc8RJt%2FIGuDLHJbpgNjZcRPrV5G0d%2BfeO2n%2F3uy2Bxmh%2BQoKt%2F3m9xLE5SpVjjF5U%2FwjhAWQbZZPDdc8rMJ7otY0GOqYB%2Fg32ebRx9MGzVLDd%2B6k8bZYrtUqolCXWBBNThlnmcVMKVY%2FXxcRh2K1y5BGXKlDivpzB9dYmRfT3B81mJ3DG4OlS5n1voC8yiHIn0qleoeyvgph8qbuFlPJlxq51MXcHA1VZtvJYe5wTrznNCmZqP4Bby56JSVVdlzMrS%2FPhdDZBRjzqAzGO8nNvIeCgznTGUd1hpFqC5rjRCshoaxJG5V%2B3lR5qXg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211206T032453Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRZJU4SVY%2F20211206%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c5b143cc04624e9d899f887cf83cf56cbf27185fa0286d447629ed2a306a7612&hash=7bbe140b0a4e1e4a8c73f2b402ff854818267ec8f356b261b5b3cb4e78e96de0&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1361841519301574&tid=spdf-a7da0fef-5b1e-49c4-a75e-040ca7ca6680&sid=ec558a9b400130484b2a9f1-fbc4af95d4adgxrqa&type=client)/Code
2020 | **MICCAI** | Adaptive context selection for polyp segmentation | [Paper](https://www.sysuhcp.com/userfiles/files/2021/02/28/8323749c38f384d5.pdf)/[Code](https://github.com/ReaFly/ACSNet) 
2020 | **MICCAI** | Mi2gan: Generative adversarial network for medical image domain adaptation using mutual information constraint | [Paper](https://arxiv.org/pdf/2007.11180.pdf)/Code 
2020 | **MICCAI** | Pranet: Parallel reverse attention network for polyp segmentation | [Paper](https://arxiv.org/pdf/2006.11392.pdf)/[Code](https://github.com/DengPingFan/PraNet) 
2020 | **MICCAI** | PolypSeg: An Efficient Context-Aware Network for Polyp Segmentation from Colonoscopy Videos | [Paper](https://link.springer.com/chapter/10.1007%2F978-3-030-59725-2_28)/Code 
2020 | **MIUA** | Polyp Segmentation with Fully Convolutional Deep Dilation Neural Network | [Paper*](https://link.springer.com/content/pdf/10.1007%2F978-3-030-39343-4.pdf)/Code
2020 | **RIVF** | Polyp Segmentation in Colonoscopy Images Using Ensembles of U-Nets with EfficientNet and Asymmetric Similarity Loss Function | [Paper](http://eprints.uet.vnu.edu.vn/eprints/id/document/3713)/Code 
2020 | **Sensors** | ABC-Net Area-Boundary Constraint Network with Dynamical Feature Selection for Colorectal Polyp Segmentation | [Paper](https://sci-hub.se/downloads/2020-08-26/00/fang2020.pdf?rand=61ab823b2d3f0?download=true)/Code 
2020 | **IEEE TMI** | Learn to Threshold: Thresholdnet with confidence-guided manifold mixup for polyp segmentation | [Paper](https://www.researchgate.net/publication/347925189_Learn_to_Threshold_ThresholdNet_With_Confidence-Guided_Manifold_Mixup_for_Polyp_Segmentation)/[Code](https://github.com/Guo-Xiaoqing/ThresholdNet) 
2019 | **IEEE Access** | Ensemble of Instance Segmentation Models for Polyp Segmentation in Colonoscopy Images | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8648333)/Code 
2019 | **CBMS** | Training Data Enhancements for Robust Polyp Segmentation in Colonoscopy Images | [Paper](https://webserver2.tecgraf.puc-rio.br/~abraposo/pubs/CBMS2019/08787526.pdf)/Code 
2019 | **EMBC** | Psi-net: Shape and boundary aware joint multi-task deep network for medical image segmentation | [Paper](https://arxiv.org/pdf/1902.04099.pdf)/[Code](https://github.com/Bala93/Multi-task-deep-network) 
2019 | **EMBC** | Polyp Segmentation using Generative Adversarial Network | [Paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8857958)/Code
2019 | **ICMLA** | Colorectal polyp segmentation by u-net with dilation convolution | [Paper](https://arxiv.org/pdf/1912.11947.pdf)/Code 
2019 | **ISM** | ResUNet++: An Advanced Architecture for Medical Image Segmentation | [Paper](https://arxiv.org/pdf/1911.07067.pdf)/[Code](https://github.com/DebeshJha/ResUNetPlusPlus-with-CRF-and-TTA)
2019 | **ISMICT** | Polyp detection and segmentation using mask r-cnn: Does a deeper feature extractor cnn always perform better? | [Paper](https://arxiv.org/pdf/1907.09180.pdf)/Code 
2019 | **MICCAI** | Selective Feature Aggregation Network with Area-Boundary Constraints for Polyp Segmentation | [Paper*](https://link.springer.com/content/pdf/10.1007%2F978-3-030-32239-7.pdf)/Code
2019 | **Nature** | U-Net – Deep Learning for Cell Counting, Detection, and Morphometry | [Paper](https://lmb.informatik.uni-freiburg.de/Publications/2019/FMBCAMBBR19/paper-U-Net.pdf)/[Code](https://lmb.informatik.uni-freiburg.de/resources/opensource/unet/)
2018 | **DLMIA** | Unet++: A nested u-net architecture for medical image segmentation | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7329239/pdf/nihms-1600717.pdf)/[Code](https://github.com/MrGiovanni/UNetPlusPlus) 
2018 | **EMBC** | Polyp segmentation in colonoscopy images using fully convolutional network | [Paper](https://arxiv.org/pdf/1802.00368.pdf)/Code 
2018 | **ICMM** | Real-Time Polyps Segmentation for Colonoscopy Video Frames Using Compressed Fully Convolutional Network | [Paper](https://www.researchgate.net/profile/Can-Udomcharoenchaikit/publication/322424455_Real-Time_Polyps_Segmentation_for_Colonoscopy_Video_Frames_Using_Compressed_Fully_Convolutional_Network/links/5ecc209a299bf1c09adf5049/Real-Time-Polyps-Segmentation-for-Colonoscopy-Video-Frames-Using-Compressed-Fully-Convolutional-Network.pdf)/Code 
2018 | **JMRR** | Towards a computed-aided diagnosis system in colonoscopy | [Paper](https://arxiv.org/pdf/2101.06040.pdf)/Code 
2018 | **Medical Robotics Res** | Automatic polyp segmentation using convolution neural networks | [Paper](https://arxiv.org/pdf/2004.10792.pdf)/Code 
2017 | **ISOP** | Fully convolutional neural networks for polyp segmentation in colonoscopy | [Paper](https://discovery.ucl.ac.uk/id/eprint/1540136/7/Rosa%20Brandao_101340F.pdf)/Code 
2017 | **SPMB** | Superpixel based segmentation and classification of polyps in wireless capsule endoscopy | [Paper](https://arxiv.org/pdf/1710.07390.pdf)/Code 
2016 | **IEEE TMI** | Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning? | [Paper](https://arxiv.org/pdf/1706.00712.pdf)/Code
2016 | **ComNet** | Advanced Algorithm for Polyp Detection Using Depth Segmentation in Colon Endoscopy | [Paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7824010)/Code
2015 | **MICCAI** | U-net: Convolutional networks for biomedical image segmentation | [Paper](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)/[Code](https://lmb.informatik.uni-freiburg.de/resources/opensource/unet/) 
2015 | **CMIG** | WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy: Validation vs. Saliency Maps from Physicians | [Paper](http://158.109.8.37/files/BSF2015.pdf)/[Code](https://polyp.grand-challenge.org/CVCClinicDB/)
2014 | **IJPRAI** | A complete system for candidate polyps detection in virtual colonoscopy | [Paper](https://arxiv.org/pdf/1209.6525.pdf)/Code 
2014 | **IEEE TMI** | Automated polyp detection in colon capsule endoscopy | [Paper](https://arxiv.org/pdf/1305.1912.pdf)/Code 
2012 | **PR** | Towards Automatic Polyp Detection with a Polyp Appearance Model | [Paper](http://refbase.cvc.uab.es/files/BSV2012a.pdf)/Code
2010 | **IEEE ICASSP** | Polyp detection in Wireless Capsule Endoscopy videos based on image segmentation and geometric feature | [Paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5495103)/Code

[Back to top](#1-preview)

## 2.2 Video Polyp Segmentation

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2022 | **MICCAI** | Semi-Supervised Spatial Temporal Attention Network for Video Polyp Segmentation | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_44)/Code
2022 | **AAAI** | TCCNet: Temporally Consistent Context-Free Network for Semi-supervised Video Polyp Segmentation | [Paper](https://www.ijcai.org/proceedings/2022/0155.pdf)/Code
2022 | **MIR** | :fire: Video Polyp Segmentation: A Deep Learning Perspective | [Paper](https://arxiv.org/pdf/2203.14291v3.pdf)/[Code](https://github.com/GewelsJI/VPS)
2021 | **MICCAI** | Progressively Normalized Self-Attention Network for Video Polyp Segmentation | [Paper](https://arxiv.org/pdf/2105.08468.pdf)/[Code](https://github.com/GewelsJI/PNS-Net)
2020 | **MICCAI** | Endoscopic Polyp Segmentation Using a Hybrid 2D/3D CNN | [Paper](https://discovery.ucl.ac.uk/id/eprint/10114066/1/Endoscopic%20polyp%20segmentation%20using%20a%20hybrid%202D-3D%20CNN.pdf)/Code

[Back to top](#1-preview)

## 2.3 Image Polyp Detection

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2022 | **JSJU** | Improving Colonoscopy Polyp Detection Rate Using Semi-Supervised Learning | [Paper](https://link.springer.com/content/pdf/10.1007/s12204-022-2519-1.pdf?pdf=inline%20link)/Code
2022 | **IJCARS** | Positive-gradient-weighted object activation mapping: visual explanation of object detector towards precise colorectal-polyp localisation | [Paper](https://link.springer.com/content/pdf/10.1007/s11548-022-02696-y.pdf)/Code
2022 | **arXiv** | Colonoscopy polyp detection with massive endoscopic images | [Paper](https://arxiv.org/pdf/2202.08730.pdf)/Code
2021 | **arXiv** | Detecting, Localising and Classifying Polyps from Colonoscopy Videos using Deep Learning | [Paper](https://arxiv.org/pdf/2101.03285.pdf)/Code 
2020 | **Scientific Data** | HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy | [Paper](https://www.nature.com/articles/s41597-020-00622-y.pdf)/[Project](https://datasets.simula.no/hyper-kvasir/)
2020 | **Scientific reports** | Real-time detection of colon polyps  during colonoscopy using deep learning: systematic validation with four independent datasets | [Paper](https://www.nature.com/articles/s41598-020-65387-1)/Code
2020 | **IEEE ISBI** | Reduce false-positive rate by active learning for automatic polyp detection in colonoscopy videos | [Paper](https://www.researchgate.net/profile/Zhe-Guo-12/publication/322563091_Automatic_polyp_recognition_from_colonoscopy_images_based_on_bag_of_visual_words/links/5f9b60a7299bf1b53e512f47/Automatic-polyp-recognition-from-colonoscopy-images-based-on-bag-of-visual-words.pdf)/Code 
2020 | **TransAI** | Artifact Detection in Endoscopic Video with Deep Convolutional Neural Networks | [Paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9253131)/Code
2019 | **IEEE Access** | Colonic Polyp Detection in Endoscopic Videos With Single Shot Detection Based Deep Convolutional Neural Network | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8731913)/Code 
2019 | **ICTAI** | An Efficient Spatial-Temporal Polyp Detection Framework for Colonoscopy Video | [Paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8995313)/Code
2018 | **arXiv** | Y-net: A deep convolutional neural network for polyp detection | [Paper](https://arxiv.org/pdf/1806.01907.pdf)/Code
2017 | **IEEE TMI** | Comparative validation of polyp detection methods in video colonoscopy: results from the MICCAI 2015 endoscopic vision challenge | [Paper](http://clok.uclan.ac.uk/17023/2/17023%20Final%20Version.pdf)/Code 
2015 | **IEEE TMI** | Automated Polyp Detection in Colonoscopy Videos Using Shape and Context Information | [Paper](https://sci-hub.se/10.1109/tmi.2015.2487997)/Code 
2009 | **Bildverarbeitung fur die Medizin** | Texturebased polyp detection in colonoscopy | [Paper](http://ftp.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-446/p346.pdf)/Code 
2009 | **Proc. SPIE** |  A comparison of blood vessel features and local binary patterns for colorectal polyp classification |  [Paper](https://www.lfb.rwth-aachen.de/files/publications/2009/GRO09a.pdf)/Code 
2007 | **IEEE ICIP** | Polyp detection in colonoscopy video using elliptical shape feature | [Paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2007/pdfs/0200465.pdf)/Code


[Back to top](#1-preview)

## 2.4 Video Polyp Detection

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2021 | **BSPC** | Real-time automatic polyp detection in colonoscopy using feature enhancement module and spatiotemporal similarity correlation unit | [Paper](https://sci-hub.se/10.1016/j.bspc.2021.102503)/Code 
2021 | **EIO** | Real-time deep learning-based colorectal polyp localization on clinical video footage achievable with a wide array of hardware configurations | [Paper](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/a-1388-6735.pdf)/Code
2021 | **MICCAI** | Multi-frame Collaboration for Effective Endoscopic Video Polyp Detection via Spatial-Temporal Feature Transformation | [Paper](https://arxiv.org/pdf/2107.03609.pdf)/[Code](https://github.com/lingyunwu14/STFT)
2020 | **IEEE ISBI** | Polyp detection in colonoscopy videos by bootstrapping via temporal consistency | [Paper](https://sci-hub.se/downloads/2020-06-30//41/ma2020.pdf?rand=61ab92dda2f6b?download=true)/Code 
2020 | **JBHI** | Improving Automatic Polyp Detection Using CNN by Exploiting Temporal Dependency in Colonoscopy Video | [Paper](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2723541/Improving+Automatic+Polyp+Detection+Using+CNN.pdf?sequence=2)/Code 
2020 | **NPJ Digital Medicine** | AI-doscopist: a real-time deep-learning-based algorithm for localising polyps in colonoscopy videos with edge computing devices | [Paper](https://www.nature.com/articles/s41746-020-0281-z.pdf)/Code
2020 | **MICCAI** | Asynchronous in Parallel Detection and Tracking (AIPDT): Real-Time Robust Polyp Detection | [Paper*](https://link.springer.com/content/pdf/10.1007%2F978-3-030-59716-0.pdf)/Code
2019 | **IEEE ISBI** | POLYP TRACKING IN VIDEO COLONOSCOPY USING OPTICAL FLOW WITH AN ON-THE-FLY TRAINED CNN | [Paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8759180)/Code
2017 | **JBHI** | Integrating Online and Offline Three-Dimensional Deep Learning for Automated Polyp Detection in Colonoscopy Videos | [Paper](http://www.cse.cuhk.edu.hk/~qdou/papers/2017/%5B2017%5D%5BJBHI%5DIntegrating%20online%20and%20offline%20three%20dimensional%20deep%20learning%20for%20automated%20polyp%20detection%20in%20colonoscopy%20videos.pdf)/Code 
2015 | **IPMI** | A Comprehensive Computer-Aided Polyp Detection System for Colonoscopy Videos | [Paper](https://link.springer.com/content/pdf/10.1007/978-3-319-19992-4_25.pdf?pdf=inline%20link)/Code
2015 | **IEEE ISBI** | Automatic polyp detection in colonoscopy videos using an ensemble of convolutional neural networks | [Paper](https://www.researchgate.net/profile/Nima-Tajbakhsh/publication/283464973_Automatic_polyp_detection_in_colonoscopy_videos_using_an_ensemble_of_convolutional_neural_networks/links/5718b4a708aed43f63221b27/Automatic-polyp-detection-in-colonoscopy-videos-using-an-ensemble-of-convolutional-neural-networks.pdf)/Code

[Back to top](#1-preview)

## 2.5 Image Polyp Classification

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2022 | **MICCAI** | FFCNet: Fourier Transform-Based Frequency Learning and Complex Convolutional Network for Colon Disease Classification | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_8)/[Code](https://github.com/soleilssss/FFCNet)
2022 | **MICCAI** | Toward Clinically Assisted Colorectal Polyp Recognition via Structured Cross-Modal Representation Consistency | [Paper](https://link.springer.com/content/pdf/10.1007/978-3-031-16437-8_14.pdf)/[Code](https://github.com/WeijieMax/CPC-Trans)
2020 | **IEEE ISBI** | Photoshopping Colonoscopy Video Frames | [Paper](https://arxiv.org/pdf/1910.10345v1.pdf)/Code
2020 | **MICCAI** | Few-Shot Anomaly Detection for Polyp Frames from Colonoscopy | [Paper](https://arxiv.org/pdf/2006.14811.pdf)/[Code](https://github.com/tianyu0207/FSAD-Net%20)
2020 | **MICCAI** | Two-Stream Deep Feature Modelling for Automated Video Endoscopy Data Analysis | [Paper](https://arxiv.org/pdf/2007.05914.pdf)/Code 
2014 | **JICARS** | Towards embedded detection of polyps in WCE images for early diagnosis of colorectal cancer | [Paper](https://hal.archives-ouvertes.fr/hal-00843459/document)/[Code](https://polyp.grand-challenge.org/EtisLarib/)

[Back to top](#1-preview)

## 2.6 Video Polyp Classification

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:
2022 | **MICCAI** | Contrastive Transformer-based Multiple Instance Learning for Weakly Supervised Polyp Frame Detection | [Paper](https://link.springer.com/content/pdf/10.1007/978-3-031-16437-8_9.pdf)/[Code](https://github.com/tianyu0207/weakly-polyp)

[Back to top](#1-preview)

## 2.7 Colonoscopy Depth Estimation

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:
2022 | **arXiv** | Task-Aware Active Learning for Endoscopic Image Analysis | [Paper](https://arxiv.org/pdf/2204.03440.pdf)/[Code](https://github.com/thetna/endo-active-learn)
2019 | **IJCARS** | Implicit domain adaptation with conditional generative adversarial networks for depth prediction in endoscopy | [Paper](https://link.springer.com/content/pdf/10.1007/s11548-019-01962-w.pdf?pdf=button%20sticky)/Code/[Project](http://cmic.cs.ucl.ac.uk/ColonoscopyDepth)

[Back to top](#1-preview)

## 2.8 Colonoscopy Deficient Coverage Detection

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:
2020 | **IEEE TMI** | Detecting Deficient Coverage in Colonoscopies | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9097918)/Code

[Back to top](#1-preview)

## 2.9 Colon Polyp Image Synthesis

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:
2018 | **IEEE Access** | Abnormal Colon Polyp Image Synthesis Using Conditional Adversarial Networks for Improved Detection Performance | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8478237)/Code

[Back to top](#1-preview)

# 3. Colonoscopy Resources

## 3.1 Image Segmentation Datasets

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2022 | **arXiv** | ERS: a novel comprehensive endoscopy image dataset for machine learning, compliant with the MST 3.0 specification | [Paper](https://arxiv.org/abs/2201.08746)/[Project](https://cvlab.eti.pg.gda.pl/publications/endoscopy-dataset)
2022 | **arXiv** | Synthetic data for unsupervised polyp segmentation | [Paper](https://arxiv.org/pdf/2202.08680.pdf)/[Code](https://github.com/enric1994/synth-colon)/[Project](https://enric1994.github.io/synth-colon)
2020 | **ICMM** | **Kvasir-SEG** - Kvasir-seg: A segmented polyp dataset | [Paper](https://arxiv.org/pdf/1911.07069.pdf)/[Code](https://datasets.simula.no/kvasir-seg/)
2017 | **JHE** | **CVC-EndoSceneStill** - A Benchmark for Endoluminal Scene Segmentation of Colonoscopy Images | [Paper](https://downloads.hindawi.com/journals/jhe/2017/4037190.pdf)/[Project](http://www.cvc.uab.es/CVC-Colon/index.php/databases/cvc-endoscenestill/)
2015 | **CMIG** | **CVC-ClinicDB/CVC-612** - WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy: Validation vs. Saliency Maps from Physicians | [Paper](http://158.109.8.37/files/BSF2015.pdf)/[Project](https://polyp.grand-challenge.org/CVCClinicDB/)
2014 | **JICARS** | **ETIS-Larib Polyp DB** - Towards embedded detection of polyps in WCE images for early diagnosis of colorectal cancer | [Paper](https://hal.archives-ouvertes.fr/hal-00843459/document)/[Project](https://polyp.grand-challenge.org/EtisLarib/)
2012 | **PR** | **CVC-ColonDB/CVC-300** - Towards Automatic Polyp Detection with a Polyp Appearance Model | [Paper](http://refbase.cvc.uab.es/files/BSV2012a.pdf)/[Project](http://mv.cvc.uab.es/projects/colon-qa/cvccolondb)
_ | _ | **PICCOLO** - PICCOLO RGB/NBI (WIDEFIELD) IMAGE COLLECTION | [Project](https://www.biobancovasco.org/en/Sample-and-data-catalog/Databases/PD178-PICCOLO-EN.html)

[Back to top](#1-preview)

## 3.2 Video Segmentation Datasets

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2017 | **EIO** | **KID Project** - KID Project: an internet-based digital video atlas of capsule endoscopy for research purposes | [Paper](https://www.thieme-connect.de/products/ejournals/pdf/10.1055/s-0043-105488.pdf)/[Project](https://mdss.uth.gr/datasets/endoscopy/kid/)
2015 | **IEEE TMI** | **ASU-Mayo** - Automated Polyp Detection in Colonoscopy Videos Using Shape and Context Information | [Paper](https://sci-hub.se/10.1109/tmi.2015.2487997)/[Project](https://polyp.grand-challenge.org/AsuMayo/) 

[Back to top](#1-preview)

## 3.3 Video Detection Datasets

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2021 | **GE** | **SUN Dataset** - Development of a computer-aided detection system for colonoscopy and a publicly accessible large colonoscopy video database (with video) | [Paper](https://pdf.sciencedirectassets.com/273305/1-s2.0-S0016510721X0003X/1-s2.0-S0016510720346551/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCoLFri%2FY2qg1dLegjpE85SraDAQgXni4AstwVHir31FQIgSZt7d3LRM%2FDWZnrG2ob5NXTOCC6qrgtukFoyETMmG60qgwQIg%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDB6thtIGUd1PJRcyGSrXA7YEHw4pAwRRjdUrDIBxA5b6lw7vwYqXu%2FkyL8wa5THz4ls%2BQ79EWG1w9zl7j9F6A9bkKlVvKCAb0oi3e03KthHn%2B8g0l4OC0qix4pb0UUwreyZgOjArd70QgeuNuGUMxagQQ0SWaQUG%2FO%2B24Zr7sqoJjCFuNyulHGwblX4JXXI9rhGeb2yWr%2FKRTmbwiCKzuerSPMtbyJGK72cZ5qWuriDfQfUoNqKp49hRkitn7ZzSrz0wayxDzK6PKgPXLzx60HsPBgz%2BcPDKsLlEKdrtnOHcpTzINtfACgeTkvm8QP5WQq9SQO4PNfOgWKMEBxkkXqNaXRlWCriDE8ikkIxTS1wg1bBzX6bbq2VXPQ1HWzoCozUUBpla1%2FNddRj3cOdWNPV1CMDZKivYiFQGuB5ARoL7ijrhNH0igSNRe2WKoerxDKdKfOVmaRm9TYwuqVN6jS%2B1nS%2Bd2yY090PHBWsBHK0ZC2ACs2gHTJdafVkDObbFKhyzU3%2B6Q3rVKCjC6Rw4sJnNz0xsDPfGKVZ%2Ffhh4QAzGdJi18NBSmADUbEEXgV8gYg6HgQvblxNqFcwrZqmCab0QYWvg6q0%2FqyJhEYcmVWEdQJr9wVCWHGNSe2%2BPFfKR51sURtmWBTCX17WNBjqlAXvCE4xPsYowWXOcK%2BWOREDfMffE6zUdWetTgKWtCjFFzqvbe%2BaeKWuwEHXQl9BuG1rERZT9fY9aEEEVE2q6W0cDvbkVuFnmXxmhpxYw6qlJisddMBTNDWrLB8llUO0sASdIj0uz5uKqE1zqL%2FLVS1CRv1fbYzdXwEyRqrHux3g%2BArKgJ5uq92X9jznB8E0mLTrlwVI7OYjZfLdkwIXvry008gFE9g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211206T024723Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQITTHIVX%2F20211206%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=9dff7d4ff54b70360044a95447978a4d661455c0e978adb1d72f44a240f04b2c&hash=96a9a5b1c7f87db2e806041f936f9262c2f0f8a853107bf2f6ef69c9e1d1db35&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0016510720346551&tid=spdf-21d9a413-2601-4aaf-8815-4d4142eeaf04&sid=ec558a9b400130484b2a9f1-fbc4af95d4adgxrqa&type=client)/[Project](http://amed8k.sundatabase.org)

[Back to top](#1-preview)

## 3.4 Video Classification Datasets

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2020 | **Scientific Data** | **HyperKvasir** - HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy | [Paper](https://www.nature.com/articles/s41597-020-00622-y.pdf)/[Project](https://datasets.simula.no/hyper-kvasir/)
2017 | **ACM MSC** | **Kvasir** - KVASIR: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3083187.3083212)/[Project](https://datasets.simula.no/kvasir/)
2017 | **IEEE TMI** | **Colonoscopic Dataset** - Computer-Aided Classification of Gastrointestinal Lesions in Regular Colonoscopy | [Paper](https://hal.archives-ouvertes.fr/hal-01291797/document)/[Project](http://www.depeca.uah.es/colonoscopy_dataset/)

[Back to top](#1-preview)

## 3.5 Colonoscopy Depth Datasets

**Yr.** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-: 
2020 | **IEEE TMI** | Detecting Deficient Coverage in Colonoscopies | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9097918)/Code/[Project](https://dl.google.com/datasets/CC20/Google-CC20-dataset.tar.gz)
2019 | **IJCARS** | Implicit domain adaptation with conditional generative adversarial networks for depth prediction in endoscopy | [Paper](https://link.springer.com/content/pdf/10.1007/s11548-019-01962-w.pdf)/[Project](http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/)

[Back to top](#1-preview)

# 4. Useful Resources 

## 4.1 Colonoscopy Related 

- [Deep Learning for Colonoscopy](https://github.com/GewelsJI/deep-learning-colonoscopy)

## 4.2 AI Conference Deadlines 

- [Acceptance Rate for AI Conferences](https://github.com/lixin4ever/Conference-Acceptance-Rate)
- [AI Conference Deadlines](https://aideadlin.es/?sub=ML,CV,NLP,RO,SP,DM)

[Back to top](#1-preview)
