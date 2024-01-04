from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math
# ---load model arc---
from model_archs.TTST_arc import TTST as net
from dataload.data import get_training_set, get_eval_set
import numpy as np
import socket
import time
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

parser.add_argument('--data_dir', type=str, default='D:/SISR/Dataset/train')
parser.add_argument('--val_dir', type=str, default='./AID-tiny/')  # val while training

parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='ttst')
parser.add_argument('--patch_size', type=int, default=64, help='Size of cropped LR image')
parser.add_argument('--residual', type=bool, default=False, help='Use global resudial or not')
parser.add_argument('--pretrained_sr', default='saved_models/ttst/xx.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='saved_models/ttst/', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='tb_logs/ttst/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
cuda = opt.gpu_mode
print(opt)

# -------- save training log ---------------
current_time = time.strftime("%H-%M-%S")
opt.save_folder = opt.save_folder + current_time + '/'
opt.log_folder = opt.log_folder + current_time + '/'
writer = SummaryWriter('./{}'.format(opt.log_folder))

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)
if not os.path.exists(opt.log_folder):
    os.makedirs(opt.log_folder)

def PSNR(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))
def checkpoint(epoch):
    model_out_path = opt.save_folder + opt.model_type + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
rewrite_print = print
def print_log(*arg):
    file_path = opt.save_folder + '/train_log.txt'
    rewrite_print(*arg)
    # 保存到文件
    rewrite_print(*arg, file=open(file_path, "a", encoding="utf-8"))

torch.cuda.manual_seed(opt.seed)

print('===> Loading training datasets')
train_set = get_training_set(opt.data_dir, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
print('===> Loading val datasets')
val_set = get_eval_set(opt.val_dir, opt.upscale_factor)
val_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

print('===> Building model ', opt.model_type)
model = net()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
print('---------- Networks architecture -------------')
print_network(model)
model = model.cuda(gpus_list[0])

if opt.pretrained:
    model_name = os.path.join(opt.pretrained_sr)
    print('load model', model_name)
    if os.path.exists(model_name):
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')
L1_criterion = nn.L1Loss()
L1_criterion = L1_criterion.cuda(gpus_list[0])
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

best_epoch = 0
best_test_psnr = 0.0
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        lr, gt = batch[0], batch[1]
        if cuda:
            gt = Variable(gt).cuda(gpus_list[0])
            lr = Variable(lr).cuda(gpus_list[0])

        optimizer.zero_grad()

        t0 = time.time()
        prediction = model(lr)
        t1 = time.time()

        loss = L1_criterion(prediction, gt)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader),
                                                                                 loss.item(), (t1 - t0)))
    print_log("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    writer.add_scalar('Avg. Loss', epoch_loss / len(training_data_loader), epoch)

    # val while training
    count = 1
    avg_psnr_predicted = 0.0
    avg_test_psnr = 0.0
    model.eval()
    for batch in val_data_loader:
        lr, gt = batch[0], batch[1]
        with torch.no_grad():
            gt = Variable(gt).cuda(gpus_list[0])
            lr = Variable(lr).cuda(gpus_list[0])
        with torch.no_grad():
            t0 = time.time()
            prediction = model(lr)
            t1 = time.time()

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)

        gt = gt.cpu()
        gt = gt.squeeze().numpy().astype(np.float32)
        gt = gt * 255.
        psnr_predicted = PSNR(prediction, gt)
        print_log("===> Processing image: %s || Timer: %.4f sec. || PSNR: %.4f dB" % (str(count), (t1 - t0), psnr_predicted))
        avg_psnr_predicted += psnr_predicted
        avg_test_psnr = avg_psnr_predicted / len(val_data_loader)
        count += 1
    if avg_test_psnr > best_test_psnr:
        best_epoch = epoch
        best_test_psnr = avg_test_psnr
    print_log("===> Epoch {} Complete: Avg. PSNR: {:.4f} Best Epoch {} Best PSNR: {:.4f}".format(epoch,
                                                                                                 avg_psnr_predicted / len(val_data_loader),
                                                                                                 best_epoch, best_test_psnr))
    writer.add_scalar('Avg. PSNR', avg_psnr_predicted / len(val_data_loader), epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if epoch % (opt.snapshots) == 0:
        checkpoint(epoch)
