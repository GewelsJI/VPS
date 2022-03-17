import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from lib.lightrfb import LightRFB
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.PNS_Module_TMI import NS_Block


class conbine_feature(nn.Module):
    def __init__(self):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(32, 16)
        self.up2_low = nn.Conv2d(24, 16, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(16)
        self.up2_act = nn.PReLU(16)
        self.refine = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.PReLU())

    def forward(self, low_fea, high_fea):
        high_fea = self.up2_high(high_fea)
        low_fea = self.up2_bn2(self.up2_low(low_fea))
        refine_feature = self.refine(self.up2_act(high_fea + low_fea))
        return refine_feature


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output


class PNSNet(nn.Module):
    def __init__(self):
        super(PNSNet, self).__init__()
        self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        self.High_RFB = LightRFB()
        self.Low_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=24)

        self.squeeze = nn.Sequential(nn.Conv2d(1024, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.decoder = conbine_feature()
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))
        self.NSB_local = NS_Block(32, radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2])  # Window = 7, 13, 7, 13 / (16, 28)
        self.NSB_global = NS_Block(32, radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4])  # Window = 19, 25, 19, 25/ (16, 28)
        print('Local: radius=[3, 3, 3, 3], dilation=[1, 2, 1, 2]')
        print('Global: radius=[3, 3, 3, 3], dilation=[3, 4, 3, 4]')
        
        print('Initializing NSB Series')

    def load_backbone(self, pretrained_dict, logger):
        model_dict = self.state_dict()
        logger.info("load_state_dict!!!")
        for k, v in pretrained_dict.items():
            if (k in model_dict):
                logger.info("load:%s" % k)
            else:
                logger.info("jump over:%s" % k)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        if len(x.shape) == 4:  # Pretrain
            origin_shape = x.shape
            x = self.feature_extractor.conv1(x)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)
            x1 = self.feature_extractor.layer1(x)
            low_feature = self.feature_extractor.layer2(x1)
            high_feature = self.feature_extractor.layer3(low_feature)

            high_feature = self.High_RFB(high_feature)
            low_feature = self.Low_RFB(low_feature)

            high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                         mode="bilinear",
                                         align_corners=False)
            out = self.decoder(low_feature, high_feature)
            out = torch.sigmoid(
                F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                              align_corners=False))
        else:  # Finetune
            origin_shape = x.shape
            x = x.view(-1, *origin_shape[2:])
            x = self.feature_extractor.conv1(x)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)
            x1 = self.feature_extractor.layer1(x)
            low_feature = self.feature_extractor.layer2(x1)
            high_feature = self.feature_extractor.layer3(low_feature)

            high_feature = self.High_RFB(high_feature)
            low_feature = self.Low_RFB(low_feature)

            high_feature = high_feature.view(*origin_shape[:2], *high_feature.shape[1:])
            low_feature = low_feature.view(*origin_shape[:2], *low_feature.shape[1:])

            high_feature_global = high_feature[:, 0, ...].unsqueeze(dim=1).repeat(1, 5, 1, 1, 1)
            high_feature_local = high_feature[:, 1:6, ...]  # torch.Size([1, 5, 32, 16, 28])

            high_feature_1 = self.NSB_global(high_feature_global, high_feature_local) + high_feature_local
            high_feature_2 = self.NSB_local(high_feature_1, high_feature_1) + high_feature_1

            high_feature = high_feature_2 + high_feature_local

            low_feature = low_feature[:, 1:6, ...]
            high_feature = high_feature.contiguous().view(-1, *high_feature.shape[2:])  # torch.Size([5, 32, 28, 42])
            low_feature = low_feature.contiguous().view(-1, *low_feature.shape[2:])  # torch.Size([5, 32, 28, 42])

            high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                         mode="bilinear",
                                         align_corners=False)

            out = self.decoder(low_feature.clone(), high_feature.clone())
            # print('out:', out.shape)
            out = torch.sigmoid(
                F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                              align_corners=False))
        return out


if __name__ == "__main__":
    a = torch.randn(1, 7, 3, 8, 14)  # .cuda()
    mobile = PNSNet()  # .cuda()
    print(mobile(a).shape)
