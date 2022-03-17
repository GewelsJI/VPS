from PIL import Image
import random
from torchvision.transforms import ToTensor as torchtotensor


################################iamge################################
class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class Random_horizontal_flip(object):
    def _horizontal_flip(self, img, label):
        # dsa
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image), "should be PIL image"
        assert isinstance(label, Image.Image), "should be PIL image"
        if random.random() < self.prob:
            return self._horizontal_flip(img, label)
        else:
            return img, label


class Random_crop_Resize(object):
    def _randomCrop(self, img, label):
        width, height = img.size
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        assert img.size == label.size, "img should have the same shape as label"
        return self._randomCrop(img, label)


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, label):
        img = img.resize((self.width, self.height), Image.BILINEAR)
        label = label.resize((self.width, self.height), Image.NEAREST)
        return img, label


class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, label):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img, label


class toTensor(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, img, label):
        img, label = self.totensor(img), self.totensor(label).long()
        return img, label


################################video################################
class Random_crop_Resize_Video(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        for img, label in zip(imgs, labels):
            img, label = self._randomCrop(img, label, x, y)
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


class Random_horizontal_flip_video(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, imgs, labels):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img = []
            res_label = []
            for img, label in zip(imgs, labels):
                img, label = self._horizontal_flip(img, label)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            return imgs, labels


class Resize_video(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
            res_label.append(label.resize((self.width, self.height), Image.NEAREST))
        return res_img, res_label


class Normalize_video(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, imgs, labels):
        res_img = []
        for img in imgs:
            for i in range(3):
                img[:, :, i] -= float(self.mean[i])
            for i in range(3):
                img[:, :, i] /= float(self.std[i])
            res_img.append(img)
        return res_img, labels


class toTensor_video(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            img, label = self.totensor(img), self.totensor(label).long()
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label