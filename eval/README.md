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

    @article{ji2022video,
      title={Video polyp segmentation: A deep learning perspective},
      author={Ji, Ge-Peng and Xiao, Guobao and Chou, Yu-Cheng and Fan, Deng-Ping and Zhao, Kai and Chen, Geng and Van Gool, Luc},
      journal={Machine Intelligence Research},
      volume={19},
      number={6},
      pages={531--549},
      year={2022},
      publisher={Springer}
    }


    @inproceedings{ji2021progressively,
      title={Progressively normalized self-attention network for video polyp segmentation},
      author={Ji, Ge-Peng and Chou, Yu-Cheng and Fan, Deng-Ping and Chen, Geng and Fu, Huazhu and Jha, Debesh and Shao, Ling},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={142--152},
      year={2021},
      organization={Springer}
    }

    @inproceedings{fan2020pranet,
      title={Pranet: Parallel reverse attention network for polyp segmentation},
      author={Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
      booktitle={International conference on medical image computing and computer-assisted intervention},
      pages={263--273},
      year={2020},
      organization={Springer}
    }
