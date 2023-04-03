import os

import torch
from torch.utils.data import Dataset

from scripts.config import config
from lib.dataloader.preprocess import *


class VideoDataset(Dataset):
    def __init__(self, video_dataset, transform=None, time_interval=1):
        super(VideoDataset, self).__init__()
        self.time_clips = config.video_time_clips
        self.video_train_list = []

        video_root = os.path.join(config.dataset_root, video_dataset)
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')

        cls_list = os.listdir(img_root)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                ))
        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                batch_clips = []
                batch_clips.append(li[0])
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                self.video_train_list.append(batch_clips)
        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li) - 1, *(label.shape))

                IMG[idx, :, :, :] = img
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx - 1, :, :, :] = label

        return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)


def get_video_dataset():
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = VideoDataset(config.dataset, transform=trsf_main, time_interval=1)

    return train_loader


if __name__ == "__main__":
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = VideoDataset(config.dataset, transform=trsf_main, time_interval=1)
