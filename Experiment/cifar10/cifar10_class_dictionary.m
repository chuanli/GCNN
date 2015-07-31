% create class specific dictionary for cifar10 dataset
% written by Chuan Li
% 2015/07/31
% based on matconvnet from A. Vedaldi and K. Lenc

close all; clear all; clc;
path_vlfeat = '/Users/chuan/Research/library/vlfeat/';
path_matconvnet = '/Users/chuan/Research/library/matconvnet/';
run ([path_vlfeat 'toolbox/vl_setup']) ;
run ([path_matconvnet 'matlab/vl_setupnn']) ;

% load pre-trained convnet
name_imdb = '/Users/chuan/Research/library/matconvnet/data/cifar-lenet/imdb.mat';
imdb = load(name_imdb);

% parameters
class_id = 2;
patch_size = 3;
num_patches = 10000;
stride = 5;

img_start = 10000;
img_end = 20000;

% sample patches
img_size = 32;
channel = 3;

coord = [];
[coord(:, :, 1), coord(:, :, 2)] = meshgrid(1:stride:img_size - patch_size + 1, 1:stride:img_size - patch_size + 1);
coord = reshape(coord, [], 2);

dict_class = zeros(num_patches, patch_size * patch_size * channel);

count = 1;
for i_img = img_start:img_end
    
    if count > num_patches
        break;
    end
    
    if imdb.images.labels(1, i_img) ~= class_id
        continue;
    else
        im_fullres = imdb.images.data(:, :, :, i_img) + imdb.averageImage;        
        for i_patch = 1:size(coord, 1)
            patch = im_fullres(coord(i_patch, 2):coord(i_patch, 2) + patch_size - 1, coord(i_patch, 1):coord(i_patch, 1) + patch_size - 1, :);
            dict_class(count, :) = patch(:)';
            count = count + 1;
            if count > num_patches
                break;
            end
        end
    end
end

save(['dict_class_' num2str(class_id) '_size_' num2str(patch_size) '.mat'], 'dict_class');