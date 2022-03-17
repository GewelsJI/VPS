import argparse

parser = argparse.ArgumentParser()

# optimizer
parser.add_argument('--gpu_id', type=str, default='0, 1, 2, 3', help='train use gpu')
parser.add_argument('--lr_mode', type=str, default="poly")
parser.add_argument('--base_lr', type=float, default=3e-4)
parser.add_argument('--finetune_lr', type=float, default=1e-4)
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

# train schedule
parser.add_argument('--pretrain_epoches', type=int, default=100)
parser.add_argument('--finetune_epoches', type=int, default=15)
parser.add_argument('--log_inteval', type=int, default=50)

# data
parser.add_argument('--data_statistics', type=str,
                    default="utils/statistics.pth", help='The normalization statistics.')
parser.add_argument('--img_dataset', type=str,
                    default=["IVPS-TrainSet"])
parser.add_argument('--video_dataset', type=str,
                    default="TrainDataset")
parser.add_argument('--test_dataset', type=str,
                    default="TestHardDataset")
parser.add_argument('--img_dataset_root', type=str,
                    default="dataset/")
parser.add_argument('--video_dataset_root', type=str,
                    default="/home/Dataset/_Dataset/")
parser.add_argument('--size', type=tuple,
                    default=(256, 448))
parser.add_argument('--pretrain_batchsize', type=int, default=24)  # 8BS == 16G
parser.add_argument('--video_batchsize', type=int, default=24)
parser.add_argument('--video_time_clips', type=int, default=5)
parser.add_argument('--video_testset_root', type=str, default="/home/Dataset/_Dataset/")

# pretrain
parser.add_argument('--pretrain_state_dict', type=str, default=None)
# parser.add_argument('--save_path', type=str, default='snapshot/PNS_TMI_series/')
parser.add_argument('--save_path', type=str, default='snapshot/PNS_TMI_series/')

config = parser.parse_args()
