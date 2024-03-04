'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import os.path as osp
import cv2
import glob
import logging
from datetime import datetime
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
from PIL import Image
import scipy.ndimage
import numpy as np
import scipy.special
import math
import lpips
import torch
from skimage.transform import resize
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def img2tensor(image):
    # image = cv2.normalize(image, None, -1, 1, cv2.NORM_MINMAX)
    # change dimension of a tensor object into a numpy array
    return torch.Tensor((image)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

# 批量计算PSNR和SSIM
def main():
    # Configurations
    crop = False  # 是否裁边
    crop_border = 4  # 左右裁剪4个像素,256*256变为248*248
    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = 'D:\Real_world_SR/fig/real_world_fig\BICUBIC'  # GT 图像路径（不要包含中文）
    folder_test = 'D:\Real_world_SR/baseline_res\iso\DOTA\CMDSR'  # 所要计算的结果路径

    subfolder_l = sorted(glob.glob(osp.join(folder_test, '*')))  # 子文件夹路径的list，即
    # subfolder_GT_l = sorted(glob.glob(osp.join(folder_GT, '*')))

    # subfolder_l = [c.replace('sig0.2_','') for c in subfolder_l]
    print(subfolder_l)
    # print(subfolder_GT_l)

    subfolder_name_l = []  # 子文件夹名
    save_logger_folder = folder_test
    setup_logger('base', save_logger_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    All_test_PSNR = []
    All_test_SSIM = []  # 保存每个clip的平均psnr，用来求最终测试集的psnr

    # for each subfolder-video clip
    for subfolder in subfolder_l:
        PSNR_all = []
        SSIM_all = []  # 保存每个clip中每张图的psnr

        subfolder_name = osp.basename(subfolder)  # 000,001...
        print(subfolder_name)
        subfolder_name_l.append(subfolder_name)  # [000, 001, ..., 011]

        img_list = sorted(glob.glob(osp.join(folder_GT, '*')))  # 真值图像路径列表

        for i, img_path in enumerate(img_list):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            print(base_name, img_path, os.path.join(folder_test, subfolder_name, base_name + suffix + '.png'))
            im_GT = cv2.imread(img_path) / 255.
            im_Gen = cv2.imread(os.path.join(folder_test, subfolder_name, base_name + suffix + '.png')) / 255.

            if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                im_GT_in = bgr2ycbcr(im_GT)
                im_Gen_in = bgr2ycbcr(im_Gen)
            else:
                im_GT_in = im_GT
                im_Gen_in = im_Gen

            # crop borders
            if crop:
                if im_GT_in.ndim == 3:
                    cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                    cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
                elif im_GT_in.ndim == 2:
                    cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
                else:
                    raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))
            else:
                cropped_GT = im_GT_in
                cropped_Gen = im_Gen_in
            PSNR = loss_fn_alex.forward(img2tensor(cropped_Gen), img2tensor(cropped_GT), normalize=True)
            PSNR = PSNR[0][0][0][0].detach().numpy()

            cropped_Gen = bgr2ycbcr(cropped_Gen)
            SSIM = niqe(cropped_Gen)

            # print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
            #     i + 1, base_name, PSNR, SSIM))
            logger.info('Folder {} {:3d} - {:25}. \tLPIPS: {:.6f} , \tNIQE: {:.6f}'.format(subfolder_name,
                i + 1, base_name, PSNR, SSIM))

            PSNR_all.append(PSNR)
            SSIM_all.append(SSIM)
        # print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        #     sum(PSNR_all) / len(PSNR_all),
        #     sum(SSIM_all) / len(SSIM_all)))
        logger.info('Folder {}-Average: LPIPS: {:.6f} , NIQE: {:.6f}'.format(subfolder_name,
            sum(PSNR_all) / len(PSNR_all),
            sum(SSIM_all) / len(SSIM_all)))
        All_test_PSNR.append(sum(PSNR_all) / len(PSNR_all))
        All_test_SSIM.append(sum(SSIM_all) / len(SSIM_all))

    logger.info('All Folder Average: LPIPS: {:.8f} , NIQE: {:.6f}'.format(sum(All_test_PSNR) / len(All_test_PSNR), sum(All_test_SSIM) / len(All_test_SSIM)))

loss_fn_alex = lpips.LPIPS(net='alex', spatial=False, use_dropout=True) # best forward scores
# lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
def calculate_correlation_coefficient(img0, img1):
    img0 = img0.reshape(img0.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
    img1 = img1.reshape(img1.size, order='C')
    return np.corrcoef(img0, img1)[0, 1]

def calculate_rmse(img1,img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return math.sqrt(mse)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm) ** 2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl) * (gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                     alpha1, N1, bl1, br1,  # (V)
                     alpha2, N2, bl2, br2,  # (H)
                     alpha3, N3, bl3, bl3,  # (D1)
                     alpha4, N4, bl4, bl4,  # (D2)
                     ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    # img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')
    size_x, size_y = img.shape
    img2 = np.array(Image.fromarray(img).resize((size_x//2, size_y//2)))

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

    return feats


def niqe(inputImgData):
    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov + sample_cov) / 2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score


if __name__ == '__main__':
    main()