# VPS Evaluation Toolbox

This toolbox is used to evaluate the performance of video polyp segmentation task.

# Usage

- Prerequisites of environment:

    ```bash
     python -m pip install opencv-python tdqm prettytable scikit-learn
  ```

- Running the evaluation:

    ```bash
    sh ./eval.sh
  ```
    By running the script, results of all models on SUN-SEG dataset will be evaluated simultaneously. If you want to evaluate the specific models, please modify the `$MODEL_NAMES`variable in `eval.sh` which is corresponding to the argument `--model_lst`. Note that the modified model name should be the same to the folder name under `./data/Pred/`.
    In `vps_evaluator.py`, you can specify `--metric_list` to decide the applying metrics. `--txt_name` denotes the folder name of evaluation result. `--data_lst` and `--check_integrity` represent the used dataset and the integrity examination of result maps and ground truth. 
    

# Citation

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