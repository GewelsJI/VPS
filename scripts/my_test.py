import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Resize

from config import config
from lib.module.PNSPlusNetwork import PNSNet as Network


def safe_save(img, save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    img.save(save_path)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img


class Test_Dataset(Dataset):
    def __init__(self, root, testset):
        time_interval = 1

        self.time_clips = config.video_time_clips
        self.video_test_list = []

        video_root = os.path.join(root, testset, 'Frame')
        cls_list = os.listdir(video_root)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []
            cls_path = os.path.join(video_root, cls)
            tmp_list = os.listdir(cls_path)

            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[
                        0])))

            for filename in tmp_list:
                self.video_filelist[cls].append(os.path.join(cls_path, filename))

        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            begin = 0  # change for inference from first frame
            while begin < len(li):
                if len(li) - begin - 1 < self.time_clips:
                    begin = len(li) - self.time_clips
                batch_clips = []
                batch_clips.append(li[0])
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                begin += self.time_clips
                self.video_test_list.append(batch_clips)

        self.img_transform = Compose([
            Resize((config.size[0], config.size[1]), Image.BILINEAR),
            ToTensor(),
            Normalize([0.4732661, 0.44874457, 0.3948762],
                      [0.22674961, 0.22012031, 0.2238305])
        ])

    def __getitem__(self, idx):
        img_path_li = self.video_test_list[idx]
        IMG = None
        img_li = []
        for idx, img_path in enumerate(img_path_li):
            img = Image.open(img_path).convert('RGB')
            img_li.append(self.img_transform(img))
        for idx, img in enumerate(img_li):
            if IMG is not None:
                IMG[idx, :, :, :] = img
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                IMG[idx, :, :, :] = img
        return IMG, img_path_li

    def __len__(self):
        return len(self.video_test_list)


class AutoTest:
    def __init__(self, test_dataset, data_root, model_path):
        assert isinstance(test_dataset, list), "error"
        self.data_root = data_root
        self.test_dataset = test_dataset
        self.dataloader = {}
        for dst in self.test_dataset:
            self.dataloader[dst] = DataLoader(Test_Dataset(data_root, dst), batch_size=1, shuffle=False, num_workers=8)
        print('Load checkpoint:', model_path)
        self.model = Network().cuda()
        new_state = {}
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        for key, value in state_dict.items():
            new_state[key.replace('module.', '')] = value

        self.tag_dir = 'res/'+model_path.split('/')[-3]+'_'+model_path.split('/')[-2]+'/'
        self.model.load_state_dict(new_state)
        self.model.eval()

    def test(self):
        with torch.no_grad():
            for dst in self.test_dataset:
                for img, path_li in tqdm(self.dataloader[dst], desc="test:%s" % dst):
                    result = self.model(img.cuda())
                    for res, path in zip(result, path_li[1:]):
                        npres = res.squeeze().cpu().numpy()
                        safe_save(Image.fromarray((npres * 255).astype(np.uint8)),
                                  path[0].replace(self.data_root, self.tag_dir).replace(".jpg", ".png").replace('Frame',
                                                                                                                ''))


if __name__ == "__main__":

    at = AutoTest(['TestEasyDataset/Seen', 'TestHardDataset/Seen', 'TestEasyDataset/Unseen', 'TestHardDataset/Unseen'],
                  config.video_testset_root,
                  "snapshot/PNSPlus/epoch_15/PNSPlus.pth")
    at.test()

