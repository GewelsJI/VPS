import os
from torch.utils.data import Dataset
from utils.preprocess import *
from torchvision import transforms

from PIL import Image
import torch
from config import config
import glob

# pretrain dataset
class Pretrain(Dataset):
    def __init__(self, img_dataset_list, transform):
        data_dir = config.video_dataset_root + img_dataset_list
        gt_path = glob.glob(data_dir + '/GT/*/*.png')
        img_path = glob.glob(data_dir + '/Frame/*/*.jpg')

        self.file_list = [(img, gt) for img, gt in zip(img_path, gt_path)]

        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_path, label_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        img, label = self._process(img, label)
        return img, label

    def _process(self, img, label):
        img, label = self.img_label_transform(img, label)
        return img, label

    def __len__(self):
        return len(self.file_list)


def get_pretrain_dataset():
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize(config.size[0], config.size[1]),
        Random_crop_Resize(15),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    train_loader = Pretrain(config.video_dataset, transform=trsf_main)

    return train_loader


# finetune dataset
class VideoDataset(Dataset):
    def __init__(self, video_dataset, transform=None, time_interval=1):
        super(VideoDataset, self).__init__()
        self.time_clips = config.video_time_clips
        self.video_train_list = []

        video_root = os.path.join(config.video_dataset_root, video_dataset)
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
            if IMG is not None:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
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
    train_loader = VideoDataset(config.video_dataset, transform=trsf_main, time_interval=1)

    return train_loader


# TMI finetune dataset
class TMIVideoDataset(Dataset):
    def __init__(self, video_dataset, transform=None, time_interval=1):
        super(TMIVideoDataset, self).__init__()
        self.time_clips = config.video_time_clips
        self.video_train_list = []

        video_root = os.path.join(config.video_dataset_root, video_dataset)
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
        FIRST = None
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
                # FIRST = torch.zeros(1, *(img.shape))
                # FIRST[idx, :, :, :] = img

                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li) - 1, *(label.shape))

                IMG[idx, :, :, :] = img
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx - 1, :, :, :] = label

        # return FIRST, IMG, LABEL
        return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)


def get_tmi_video_dataset():
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = TMIVideoDataset(config.video_dataset, transform=trsf_main, time_interval=1)

    return train_loader


class TMITestVideoDataset(Dataset):
    def __init__(self, testset):
        time_interval = 1

        self.video_filelist = testset
        self.time_clips = config.video_time_clips
        self.video_test_list = []

        video_root = os.path.join(config.video_dataset_root, testset)
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')

        cls_list = os.listdir(img_root)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []
            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            tmp_list = os.listdir(cls_img_path)

            try:
                tmp_list.sort(
                    key=lambda name: int(name.rstrip('.jpg'))
                )
            except:
                tmp_list.sort(key=lambda name: (

                    int(name.split('_a')[1].split('_')[0]),
                    int(name.split('_image')[1].split('.jpg')[
                            0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                ))

        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            begin = 0  # change for inference from frist frame
            while begin < len(li):
                if len(li) - 1 - begin < self.time_clips:
                    begin = len(li) - self.time_clips
                batch_clips = []
                batch_clips.append(li[0])
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                begin += self.time_clips
                self.video_test_list.append(batch_clips)

        self.img_transform = transforms.Compose([
            transforms.Resize((config.size[0], config.size[1]), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.4732661, 0.44874457, 0.3948762],
                      [0.22674961, 0.22012031, 0.2238305])
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize((config.size[0]//4, config.size[1]//4), Image.BILINEAR),
            transforms.ToTensor()
        ])

    def _process_frame(self, img):
        img = self.img_transform(img)
        return img

    def __getitem__(self, idx):
        img_label_li = self.video_test_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')

            img_li.append(self.img_transform(img))
            label_li.append(self.label_transform(label))

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
        return len(self.video_test_list)


def get_tmi_test_video_dataset():

    eval_loader = TMITestVideoDataset(config.test_dataset)

    return eval_loader


if __name__ == "__main__":
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = VideoDataset(config.video_dataset_list, transform=trsf_main, time_interval=1)
