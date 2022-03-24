# -*- coding: utf-8 -*-
# @Time     : 2022/03/14
# @Author   : Johnson-Chou
# @Email    : johnson111788@gmail.com
# @FileName : vps_evaluator.py

import glob
import os
import cv2
import argparse
from tqdm import tqdm
import prettytable as pt
import numpy as np

def get_competitors(root):
    for model_name in os.listdir(root):
        print('\'{}\''.format(model_name), end=', ')


def evaluator(gt_pth_lst, pred_pth_lst, metrics):
    module_map_name = {"Smeasure": "Smeasure", "wFmeasure": "WeightedFmeasure", "MAE": "MAE",
                       "adpEm": "Emeasure", "meanEm": "Emeasure", "maxEm": "Emeasure",
                       "adpFm": "Fmeasure", "meanFm": "Fmeasure", "maxFm": "Fmeasure",
                       "meanSen": "Medical", "maxSen": "Medical", "meanSpe": "Medical", "maxSpe": "Medical",
                       "meanDice": "Medical", "maxDice": "Medical", "meanIoU": "Medical", "maxIoU": "Medical"}
    res, metric_module = {}, {}
    metric_module_list = [module_map_name[metric] for metric in metrics]
    metric_module_list = list(set(metric_module_list))

    # define measures
    for metric_module_name in metric_module_list:
        metric_module[metric_module_name] = getattr(__import__("metrics", fromlist=[metric_module_name]),
                                                    metric_module_name)(length=len(gt_pth_lst))

    assert len(gt_pth_lst) == len(pred_pth_lst)

    # evaluator
    for idx in tqdm(range(len(gt_pth_lst))):
        gt_pth = gt_pth_lst[idx]
        pred_pth = pred_pth_lst[idx]

        assert os.path.isfile(gt_pth) and os.path.isfile(pred_pth)

        pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)
        gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)

        # ensure the shape of prediction is matched to gt
        if not gt_ary.shape == pred_ary.shape:
            pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        for module in metric_module.values():
            module.step(pred=pred_ary, gt=gt_ary, idx=idx)

    for metric in metrics:
        module = metric_module[module_map_name[metric]]
        res[metric] = module.get_results()[metric]

    return res


def eval_engine_vps(opt, txt_save_path):

    # evaluation for whole dataset
    for _data_name in opt.data_lst[0]:
        print('#' * 20, 'Current Dataset:', _data_name, '#' * 20)
        filename = os.path.join(txt_save_path, '{}_eval.txt'.format(_data_name))

        with open(filename, 'w+') as file_to_write:

            # initial settings for PrettyTable
            tb = pt.PrettyTable()
            names = ["Dataset", "Method"]
            names.extend(opt.metric_list)
            tb.field_names = names

            # iter each method for current dataset
            for _model_name in opt.model_lst[0]:
                print('#' * 10, 'Current Method:', _model_name, '#' * 10)

                gt_src = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_src = os.path.join(opt.pred_root, _model_name, _data_name)

                # get the sequence list for current dataset
                case_list = os.listdir(gt_src)
                mean_case_score_list, max_case_score_list = [], []
                # iter each video frame for current method-dataset
                for case in case_list:
                    case_gt_name_list = glob.glob(gt_src + '/{}/*.png'.format(case))

                    try:
                        case_gt_name_list.sort(
                            key=lambda name: (int(name.split('/')[-2]), int(name.split('/')[-1].rstrip('.png')))
                        )
                    except:
                        case_gt_name_list.sort(
                            key=lambda name: (int(name.split("/")[-2].split('case')[1].split('_')[0]),
                                              0 if not len(name.split('/')[-2].split('_')) > 1 else int(
                                                  name.split('/')[-2].split('_')[1]),
                                              int(name.split('/')[-1].split('_a')[1].split('_')[0]),
                                              int(name.split('/')[-1].split('_image')[1].split('.png')[
                                                      0])))

                    # for fair comparison, we remove the first frame and last frame in the video suggested by reference: Shifting More Attention to Video Salient Object Detection
                    # https://github.com/DengPingFan/DAVSOD/blob/master/EvaluateTool/main.m
                    case_gt_name_list = case_gt_name_list[1:-1]
                    case_pred_name_list = [gt.replace(gt_src, pred_src) for gt in case_gt_name_list]

                    result = evaluator(
                        gt_pth_lst=case_gt_name_list,
                        pred_pth_lst=case_pred_name_list,
                        metrics=opt.metric_list
                    )
                    mean_score_ind, max_score_ind = [], []
                    mean_score_list, max_score_list = [], []
                    for i, (name, value) in enumerate(result.items()):
                        if 'max' in name or 'mean' in name:
                            if 'max' in name:
                                max_score_list.append(value)
                                max_score_ind.append(i)
                            else:
                                mean_score_list.append(value)
                                mean_score_ind.append(i)
                        else:
                            mean_score_list.append([value]*256)
                            mean_score_ind.append(i)

                    # calculate all the metrics at frame-level
                    max_case_score_list.append(max_score_list)
                    mean_case_score_list.append(mean_score_list)

                # calculate all the metrics at sequence-level
                max_case_score_list = np.mean(np.array(max_case_score_list), axis=0)
                mean_case_score_list = np.mean(np.array(mean_case_score_list), axis=0)
                case_score_list = []
                for index in range(len(opt.metric_list)):
                    real_max_index = np.where(np.array(max_score_ind) == index)
                    real_mean_index = np.where(np.array(mean_score_ind) == index)
                    if len(real_max_index[0]) > 0:
                        case_score_list.append(max_case_score_list[real_max_index[0]].max().round(3))
                    else:
                        case_score_list.append(mean_case_score_list[real_mean_index[0]].mean().round(3))


                final_score_list = ['{:.3f}'.format(case) for case in case_score_list]
                tb.add_row([_data_name, _model_name] + list(final_score_list))
            print(tb)
            file_to_write.write(str(tb))
            file_to_write.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='custom your ground-truth root',
        default='../data/SUN-SEG-Annotation/')
    parser.add_argument(
        '--pred_root', type=str, help='custom your prediction root',
        default='../data/Pred/')
    parser.add_argument(
        '--metric_list', type=list, help='set the evaluating metrics',
        default=['Smeasure', 'maxEm', 'wFmeasure', 'maxFm', 'maxDice', 'maxIoU'],
        choices=["Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm", "meanFm", "maxFm",
                 "meanSen", "maxSen", "meanSpe", "maxSpe", "meanDice", "maxDice", "meanIoU", "maxIoU"])
    parser.add_argument(
        '--data_lst', type=str, help='set the dataset what you wanna to test',
        nargs='+', action='append',
        choices=['CVC-ColonDB-300', 'CVC-ClinicDB-612', 'TestEasyDataset', 'TestHardDataset', 'TestEasy'])
    parser.add_argument(
        '--model_lst', type=str, help='candidate competitors',
        nargs='+', action='append',
        choices=['2015-MICCAI-UNet', '2018-TMI-UNet++', '2020-MICCAI-ACSNet', '2020-MICCAI-PraNet',
                 '2021-MICCAI-SANet', '2019-TPAMI-COSNet', '2020-AAAI-PCSA', '2020-MICCAI-23DCNN', '2020-TIP-MATNet',
                 '2021-ICCV-DCFNet', '2021-ICCV-FSNet', '2021-MICCAI-PNSNet', '2021-NIPS-AMD', '2022-TMI-PNSPlus'])
    parser.add_argument(
        '--txt_name', type=str, help='logging root',
        default='Benchmark')
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=True)
    opt = parser.parse_args()

    txt_save_path = './eval-result/{}/'.format(opt.txt_name)
    os.makedirs(txt_save_path, exist_ok=True)

    # TODO: check the integrity of each candidates @Johnson-Chou
    if opt.check_integrity:
        for _data_name in opt.data_lst[0]:
            for _model_name in opt.model_lst[0]:
                gt_pth = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_pth = os.path.join(opt.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    print('The {} Dataset of {} Model is not matching to the ground-truth'.format(_data_name,
                                                                                                  _model_name))
        # raise Exception('check done')
    else:
        print('>>> Skip check the integrity of each candidates ...')

    # start eval engine
    eval_engine_vps(opt, txt_save_path)
