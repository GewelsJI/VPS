import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from lib.lightrfb import LightRFB
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.PNS_Module import NS_Block
from utils.utils import heatmap

class conbine_feature(nn.Module):
    def __init__(self):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(64, 16)
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
        print('in0', in0.shape)
        # print(np.unique(in0.detach().cpu().numpy()))
        # RuntimeError: CUDA error: an illegal memory access was encountered
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        heatmap(b1[0], 'b1')
        heatmap(b2[0], 'b2')
        print('b1', b1.shape, 'b2', b2.shape)  # b1 torch.Size([5, 8, 32, 56]) b2 torch.Size([5, 8, 32, 56])
        # print(np.unique(b1.detach().cpu().numpy()))
        # RuntimeError: CUDA error: an illegal memory access was encountered
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

    def forward(self, first, x):
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
            first = first.view(-1, *origin_shape[2:])

            low_feature, high_feature = self.feature_extractor(x)

            first_low_feature, first_high_feature = self.feature_extractor(first)

            high_feature = self.High_RFB(high_feature)
            low_feature = self.Low_RFB(low_feature)

            first_high_feature = self.High_RFB(first_high_feature)
            heatmap(first_high_feature[0], 'first_high_feature')
            high_feature = high_feature.view(*origin_shape[:2], *high_feature.shape[1:])
            first_high_feature = first_high_feature.view(*origin_shape[:1], 1, *first_high_feature.shape[1:])
        
            high_feature_global = self.NSB_global(first_high_feature, high_feature) + high_feature
            high_feature_local = self.NSB_local(high_feature, high_feature) + high_feature
            print('high_feature_global', high_feature_global.shape,'high_feature_local', high_feature_local.shape)
            # return torch.cat((self[:PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems:]))
            # RuntimeError: cuda runtime error (77) : an illegal memory access was encountered at
            # /opt/conda/conda-bld/pytorch_1556653099582/work/aten/src/THC/THCCachingHostAllocator.cpp:265
            high_feature = torch.cat((high_feature_local, high_feature_global), dim=2)

            high_feature = high_feature.view(-1, *high_feature.shape[2:])  # torch.Size([7, 32, 16, 28])

            high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                         mode="bilinear",
                                         align_corners=False)

            out = self.decoder(low_feature.clone(), high_feature.clone())
            out = torch.sigmoid(
                F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                              align_corners=False))
        return out


if __name__ == "__main__":
    # Problem 1
    #   File "/home/Code/video-polyp-segmentation/lib/benchmark/video/PNS-Net/lib/PNS_Network_TMI_gp_sep.py", line 41, in forward
    #     in0 = self.conv0(input)
    #   File "/root/miniconda3/envs/PNSNet/lib/python3.6/site-packages/torch/nn/modules/module.py", line 493, in __call__
    #     result = self.forward(*input, **kwargs)
    #   File "/root/miniconda3/envs/PNSNet/lib/python3.6/site-packages/torch/nn/modules/conv.py", line 338, in forward
    #     self.padding, self.dilation, self.groups)
    # RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
    # Solved by removing cudnn.benchmark = True

    # Problem 2
    # File "/home/Code/video-polyp-segmentation/lib/benchmark/video/PNS-Net/lib/PNS_Network_TMI_gp_sep.py", line 48, in
    #     output = torch.cat([b1, b2], dim=1)
    # RuntimeError: cuda runtime error (77) : an illegal memory access was encountered at
    # /opt/conda/conda-bld/pytorch_1556653099582/work/aten/src/THC/THCCachingHostAllocator.cpp:265

    import logging
    import torch.backends.cudnn as cudnn

    model = PNSNet().cuda()
    model.load_backbone(torch.load('../snapshot/PNS_pretrain/epoch_101/PNS_Pretrain.pth', map_location=torch.device('cpu')), logging)
    model.cuda().train()

    for i in range(100):
        a = torch.randn(1, 1, 3, 256, 448).cuda()
        b = torch.randn(1, 5, 3, 256, 448).cuda()

        print(model(a, b).shape)
