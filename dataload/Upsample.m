clear;
close all;
%% 数据的批量上（下）采样
LR_clip_folder = 'D:\SISR\figures\real_exp\real-gt\GT\';

save_LR_folder = 'D:\SISR\Dataset\test\DIOR1000\LR\';

save_upsample_folder = 'D:\SISR\Dataset\test\DIOR1000\LR\';
save_bicubic_folder = 'D:\SISR\figures\real_exp\real-gt\blcubic\';

if ~exist(save_upsample_folder,'dir')
    mkdir(save_upsample_folder);
end
if ~exist(save_bicubic_folder,'dir')
    mkdir(save_bicubic_folder);
end
%%
filepath = dir(fullfile(LR_clip_folder,'*.png'));
up_scale=4;
parfor j=1:1:length(filepath)
    img_name = filepath(j).name;
    img = imread(fullfile(LR_clip_folder,filepath(j).name));
    img = im2double(img);

%     im_HR= imresize(img, 1/up_scale, 'bicubic');
    im_bic = imresize(img, up_scale, 'bicubic');
    img_name = filepath(j).name;
%     imwrite(im_HR,fullfile(save_LR_folder,img_name));
    imwrite(im_bic,fullfile(save_bicubic_folder,img_name));
end 
