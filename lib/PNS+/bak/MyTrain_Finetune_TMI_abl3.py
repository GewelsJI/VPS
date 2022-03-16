import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from config import config
from lib.PNS_Network_TMI_series_abl3 import PNSNet as Network
from utils.dataloader import get_tmi_video_dataset, get_tmi_test_video_dataset
from utils.utils import clip_gradient, adjust_lr
import numpy as np
import eval.metrics as Measure
from tqdm import tqdm

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        total_loss = F.binary_cross_entropy(pred.squeeze(), target.squeeze().float())
        return total_loss


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.cuda().train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
    
            images = images.cuda()
            gts = gts.cuda()
            
            preds = model(images)
            
            loss = CrossEntropyLoss().cuda()(preds.squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:])))
            loss.backward()

            clip_gradient(optimizer, config.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, config.finetune_epoches, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(epoch, config.finetune_epoches, i, total_step, loss.data))

                os.makedirs(os.path.join(save_path, "epoch_%d" % (epoch + 1)), exist_ok=True)
                save_root = os.path.join(save_path, "epoch_%d" % (epoch + 1))
                torch.save(model.state_dict(), os.path.join(save_root, "PNS_Finetune.pth"))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.finetune_epoches, loss_all))


    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def evaluate(eval_loader, model, epoch, save_path):
    """
    validation function
    """
    global best_metric_dict, best_score, best_epoch

    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()

    metrics_dict = dict()

    model.eval()
    with torch.no_grad():

        for i, (images, gts) in tqdm(enumerate(eval_loader, start=1)):
            images = images.cuda()

            gts = np.asarray(gts, np.float32).squeeze()

            images = images.cuda()

            out = model(images)
            # print(out.shape, np.unique(gts))
            for idx in range(gts.shape[0]):
                res = F.upsample(out[idx].unsqueeze(dim=0), size=gts[idx].shape, mode='bilinear', align_corners=False)
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                WFM.step(pred=res, gt=gts[idx])
                SM.step(pred=res, gt=gts[idx])
                EM.step(pred=res, gt=gts[idx])

        metrics_dict.update(wFm=WFM.get_results()['wfm'])
        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(adpEm=EM.get_results()['em']['adp'])

        cur_score = metrics_dict['wFm'] + metrics_dict['Sm'] + metrics_dict['adpEm']

        if epoch == 0:
            best_score = cur_score
            best_metric_dict = metrics_dict
            print('[Cur Epoch: {}] Metrics (wFm={}, Sm={}, adpEm={})'.format(
                epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['adpEm']))
            logging.info('[Cur Epoch: {}] Metrics (wFm={}, Sm={}, adpEm={})'.format(
                epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['adpEm']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print('[Cur Epoch: {}] Metrics (wFm={}, Sm={}, adpEm={})\n[Best Epoch: {}] Metrics (wFm={}, Sm={}, adpEm={})'.format(
                epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['adpEm'],
                best_epoch, best_metric_dict['wFm'], best_metric_dict['Sm'], best_metric_dict['adpEm']))
            logging.info('[Cur Epoch: {}] Metrics (wFm={}, Sm={}, adpEm={})\n[Best Epoch:{}] Metrics (wFm={}, Sm={}, adpEm={})'.format(
                epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['adpEm'],
                best_epoch, best_metric_dict['wFm'], best_metric_dict['Sm'], best_metric_dict['adpEm']))


if __name__ == '__main__':

    config.save_path = config.save_path.replace('PNS_TMI_series', 'PNS_TMI_series_abla3')

    model = Network().cuda()

    if config.pretrain_state_dict is not None:
        model.load_backbone(torch.load(config.pretrain_state_dict, map_location=torch.device('cpu')), logging)
        print('load model from ', config.pretrain_state_dict)


    if config.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif config.gpu_id == '0, 1':
        model = nn.DataParallel(model)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        print('USE GPU 0 and 1')
    elif config.gpu_id == '2, 3':
        model = nn.DataParallel(model)
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        print('USE GPU 2 and 3')
    elif config.gpu_id == '0, 1, 2, 3':
        model = nn.DataParallel(model)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
        print('USE GPU 0, 1, 2 and 3')

    cudnn.benchmark = True

    base_params = [params for name, params in model.named_parameters() if ("temporal_high" in name)]
    finetune_params = [params for name, params in model.named_parameters() if ("temporal_high" not in name)]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': config.base_lr, 'weight_decay': 1e-4, 'name': "base_params"},
        {'params': finetune_params, 'lr': config.finetune_lr, 'weight_decay': 1e-4, 'name': 'finetune_params'}])

    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader =get_tmi_video_dataset()
    train_loader = data.DataLoader(dataset=train_loader,
                                   batch_size=config.video_batchsize,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=False
                                   )
    logging.info('Train on {}'.format(config.video_dataset))
    print('Train on {}'.format(config.video_dataset))
    eval_loader =get_tmi_test_video_dataset()
    eval_loader = data.DataLoader(dataset=eval_loader,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=False
                                   )
    logging.info('Validate on {}'.format(config.test_dataset))
    print('Validate on {}'.format(config.test_dataset))
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    print("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(config.finetune_epoches, config.base_lr, config.video_batchsize, config.size, config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch))
    print('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(config.finetune_epoches, config.base_lr, config.video_batchsize, config.size, config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch))
    step = 0
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(config.finetune_epoches):
        cur_lr = adjust_lr(optimizer, config.base_lr, epoch, config.decay_rate, config.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        # if epoch == 0 or (epoch+1) % 10 == 0:
        #     evaluate(eval_loader, model, epoch, save_path)
