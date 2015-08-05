% deepdream with mrf prior
% written by Chuan Li
% 2015/07/31
% based on matconvnet from A. Vedaldi and K. Lenc
% close all; clear all; clc;
% path_vlfeat = '/Users/chuan/Research/library/vlfeat/';
% path_matconvnet = '/Users/chuan/Research/library/matconvnet/';
% run ([path_vlfeat 'toolbox/vl_setup']) ;
% run ([path_matconvnet 'matlab/vl_setupnn']) ;
% addpath('../../Misc/deep-goggle');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Setup experiments
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% name_model = '/Users/chuan/Research/library/matconvnet/data/imagenet/imagenet-vgg-verydeep-19.mat';
% name_googledream = '/Users/chuan/Research/Dataset/data/deepdream/';
% name_output = '/Users/chuan/Research/Dataset/data/deepdream/mrf/';
% format = '.png';
% if(~exist(name_model, 'file'))
%     warning('Pretrained model not found\n');
% end
% net = load(name_model);

name_img = ['/Users/chuan/Research/Dataset/data/deepdream/bath.jpg'];
name_dream = ['/Users/chuan/Research/Dataset/data/deepdream/mrf/output_oct_3_10.png'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.num_layers = 36;
opts.octave_num = 3;
opts.octave_scale = 2;
opts.num_iter = 10;
opts.lr = 1.5; % learning rate

opts.up_scale = 1;
opts.average = 128; % image mean
opts.bound = 255 - opts.average;
opts.channel = 3;

opts.patch_mrf_size = 3;
opts.stride_mrf_source = 1;
opts.stride_mrf_target = 1;
opts.lambdaMRF = 0.5; % weight for MRF constraint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im_fullres = imread(name_img);
dream_fullres = imread(name_dream);
net.normalization.averageImage = single(ones(size(im_fullres)) * opts.average);
net.normalization.imageSize = size(im_fullres);
opts.normalize = get_cnn_normalize(net.normalization) ;
opts.denormalize = get_cnn_denormalize(net.normalization) ;
opts.imgSize = net.normalization.imageSize;
net.layers = net.layers(1, 1:opts.num_layers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% render
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sel_feature = 14;
sel_layer = 37;
scaler2fullres = 16;

x_ini = im_fullres;
dream_ini = dream_fullres;

im_fullres = imread(name_img);
dream_fullres = imread(name_dream);

x_ini = opts.normalize(x_ini);
dream_ini = opts.normalize(dream_ini);

res_x = vl_simplenn(net, x_ini);
res_dream = vl_simplenn(net, dream_ini);

coord_mrf_source = [];
[coord_mrf_source(:, :, 1), coord_mrf_source(:, :, 2)] = meshgrid([(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_source:size(res_x(sel_layer).x(:, :, 1), 2) - (opts.patch_mrf_size - 1)/2 - 1, size(res_x(sel_layer).x(:, :, 1), 2) - (opts.patch_mrf_size - 1)/2], ...
    [(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_source:size(res_x(sel_layer).x(:, :, 1), 1) - (opts.patch_mrf_size - 1)/2 - 1, size(res_x(sel_layer).x(:, :, 1), 1) - (opts.patch_mrf_size - 1)/2]);
coord_mrf_source = reshape(coord_mrf_source, [], 2);
coord_mrf_source_fullres =  coord_mrf_source * scaler2fullres - scaler2fullres/2 + 1;
coord_mrf_target_fullres = coord_mrf_source_fullres;

[row, col] = ind2sub(size(res_x(sel_layer).x(:, :, 1)), sel_feature);

feature_x = reshape(res_x(sel_layer).x(row, col, :), 1, []);
im_x = 0 * im_fullres;
im_x(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :) = ...
    im_fullres(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :);

feature_dream = reshape(res(sel_layer).x(row, col, :), 1, []);
im_dream = 0 * im_fullres;
im_dream(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :) = ...
    dream_fullres(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :);

close all;
figure;
imshow([im_x 255 * ones(size(im_x, 1), 5, 3) im_dream]);

figure;
hold on;
plot([1:size(feature_x, 2)], feature_x, 'r');
plot([1:size(feature_dream, 2)], feature_dream, 'b');
axis([0 size(feature_x, 2) -500 500]);


