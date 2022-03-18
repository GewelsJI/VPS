import os
import time
import torch
import numpy as np
from ptflops import get_model_complexity_info


def computeTime(model, inputs, device='cuda'):
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): %.4f, FPS:%d'%(np.mean(time_spent),1*1//np.mean(time_spent)))
    return 1*1//np.mean(time_spent)


if __name__=="__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    torch.backends.cudnn.benchmark = True

    from lib.module.PNSPlusNetwork import PNSNet as Network

    model = Network().cuda()
    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (6, 3, 256, 448), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
    inputs = torch.randn(1, 6, 3, 256, 448)

    print(str(params) + '\t' + str(macs))
    computeTime(model, inputs)

