clc;close all;clear;

AID_dir = 'E:\SISR-dataset\AID\';
AID_class_name = {'Airport\','BareLand\','BaseballField\','Beach\',...
    'Bridge\','Center\','Church\','Commercial\','DenseResidential\','Desert\','Farmland\',...
    'Forest\','Industrial\','Meadow\','MediumResidential\','Mountain\','Park\','Parking\',...
    'Playground\','Pond\','Port\','RailwayStation\','Resort\','River\','School\','SparseResidential\',...
    'Square\','Stadium\','StorageTanks\','Viaduct\'};
parfor n=1:1:length(AID_class_name)
    class_folder = AID_class_name{1,n};
    jpg_list = dir([AID_dir,class_folder,'*.jpg']);

    img_num = length(jpg_list);
%     training_num = img_num/2;
%     test_num = 10;
    select_rule = 100 + 30;
    rand_num = randperm(img_num, select_rule);  %每个类别下，100的用于训练，30张用于测试

    for i=1:1:(select_rule)
        idx = rand_num(i);
        if i<=100
            img_save_folder = 'D:\SISR\Dataset\train\GT\';
        else
            img_save_folder = ['D:\SISR\Dataset\test\AID900\GT\',class_folder,'\'];
        end

        if ~exist(img_save_folder,'dir')
            mkdir(img_save_folder);
        end

        img = imread([AID_dir,class_folder,jpg_list(idx).name]);
        img = im2double(img);
        img = img(44:555, 44:555, :)
%         img = imresize(img, [512, 512], 'bicubic');
        png_name = replace(jpg_list(idx).name,'jpg','png')
        imwrite(img,[img_save_folder,png_name]);
    end
end