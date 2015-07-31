close all; clear all; clc;
addpath('./reconstruction'); addpath('./helpers');
run /Users/chuan/Research/library/vlfeat/toolbox/vl_setup ;
run /Users/chuan/Research/library/matconvnet/matlab/vl_setupnn ;

name_imdb = '/Users/chuan/Research/library/matconvnet/data/cifar-lenet/imdb.mat';
imdb = load(name_imdb);

num_img = 10000;
patch_size = 3;
img_size = 32;
stride = 5;
class_id = 5;
num_patches = 10000;

x = [];
[x(:, :, 1), x(:, :, 2)] = meshgrid(1:stride:img_size - patch_size + 1, 1:stride:img_size - patch_size + 1);
x = reshape(x, [], 2);

dict_class = zeros(num_patches, patch_size * patch_size * 3);

count = 1;
for i_img = 10000:10000 + num_img
    
    if count > num_patches
        break;
    end
    
    if imdb.images.labels(1, i_img) ~= class_id
        continue;
    else
        im_fullres = imdb.images.data(:, :, :, i_img) + imdb.averageImage;        
        for i_patch = 1:size(x, 1)
            patch = im_fullres(x(i_patch, 2):x(i_patch, 2) + patch_size - 1, x(i_patch, 1):x(i_patch, 1) + patch_size - 1, :);
            dict_class(count, :) = patch(:)';
            count = count + 1;
            if count > num_patches
                break;
            end
        end
    end
end

save(['dict_class_' num2str(class_id) '_size_' num2str(patch_size) '.mat'], 'dict_class');


return;
figure;
imshow(reshape(double(dict_class(1011, :))/255, patch_size, patch_size, []));