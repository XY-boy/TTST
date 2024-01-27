from __future__ import print_function
import argparse

import os
import torch
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from os import listdir
import math
# ---load model architecture---
from model_archs.TTST_arc import TTST
import glob
import numpy as np
import socket
import time
import imageio
from PIL import Image

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

parser.add_argument('--data_dir', type=str, default='D:/SISR/Dataset/test/DIOR1000')

parser.add_argument('--model_type', type=str, default='ttst')
parser.add_argument('--pretrained_sr', default='saved_models/ttst_4x.pth', help='sr pretrained base model')
parser.add_argument('--save_folder', default='results/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
cuda = opt.gpu_mode
print(opt)

current_time = time.strftime("%H-%M-%S")
opt.save_folder = opt.save_folder + current_time + '/'

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

transform = transform.Compose([transform.ToTensor(),])
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

torch.cuda.manual_seed(opt.seed)
device = 'cuda:0'
print('===> Building model ', opt.model_type)
model = TTST()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
print('---------- Networks architecture -------------')
print_network(model)
model = model.cuda(gpus_list[0])

model_name = os.path.join(opt.pretrained_sr)
if os.path.exists(model_name):
    # model= torch.load(model_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(torch.load(model_name))
    print('Pre-trained SR model is loaded.')
else:
    print('No pre-trained model!!!!')

def eval(folder_name):
    print('===> Loading val datasets')
    LR_filename = os.path.join(opt.data_dir, 'LR') + '/' + folder_name
    LR_image = sorted(glob.glob(os.path.join(LR_filename, '*')))  # LR图像路径列表

    # test begin
    model.eval()
    for i, img_path in enumerate(LR_image):
        # lr = imageio.v2.imread(img_path)
        # lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        # lr = torch.from_numpy(lr).float().to(device).unsqueeze(0)

        lr = Image.open(img_path).convert('RGB')
        lr = transform(lr).unsqueeze(0)

        with torch.no_grad():
            t0 = time.time()
            prediction = model(lr)
            t1 = time.time()

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)
        # print(prediction.shape)
        prediction = prediction.transpose(1, 2, 0)

        print("===> Processing image: %s || Timer: %.4f sec." % (img_path, (t1 - t0)))
        save_name = os.path.splitext(os.path.basename(img_path))[0]
        save_foler = opt.save_folder + folder_name
        if not os.path.exists(save_foler):
            os.makedirs(save_foler)
        save_fn = save_foler + save_name + '.png'
        print('save image to:', save_fn)
        Image.fromarray(np.uint8(prediction)).save(save_fn)
        # print(prediction.shape)  # (512, 512, 3)
        # cv2.imwrite(save_fn, prediction, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(save_fn, cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == '__main__':
    AID_class_name = ['Airport/','BareLand/','BaseballField/','Beach/','Bridge/','Center/','Church/','Commercial/','DenseResidential/',
                      'Desert/','Farmland/','Forest/','Industrial/','Meadow/','MediumResidential/','Mountain/','Park/','Parking/','Playground/',
                      'Pond/','Port/','RailwayStation/','Resort/','River/','School/','SparseResidential/','Square/','Stadium/','StorageTanks/','Viaduct/']
    dota_class = ['']
    for folder in dota_class:
        eval(folder_name=folder)