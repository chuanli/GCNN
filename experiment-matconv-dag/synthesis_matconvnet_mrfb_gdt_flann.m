% deepdream with mrf prior
% written by Chuan Li
% 2015/07/31
% based on matconvnet from A. Vedaldi and K. Lenc
close all; clear all; clc;
path_matconvnet = '../third-party/matconvnet-dag/';
run ([path_matconvnet 'matlab/vl_setupnn']) ;
addpath('./Misc/deep-goggle');

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup experiments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
name_model = '../data/nets/imagenet-vgg-verydeep-19.mat';
name_output = '../data/output/synthesis_matconvnet_mrfb_gdt_flann/';
mkdir(name_output);
format = '.png';
if(~exist(name_model, 'file'))
    warning('Pretrained model not found\n');
end
net = load(name_model);

name_img = ['../data/input/img_ini/bath_full.jpg'];
name_texture = ['../data/input/example_mrf/Lizard_11.jpg'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.num_layers = 36;
opts.octave_num = 3;
opts.octave_scale = 2;
opts.num_iter = 20;
opts.lr = 2.5; % learning rate

opts.up_scale = 1;
opts.average = 128; % image mean
opts.bound = 255 - opts.average;
opts.channel = 3;

opts.patch_mrf_size = 3;
opts.stride_mrf_source = 5;
opts.stride_mrf_target = 1;
% opts.lambdaMRF = 0;
opts.lambdaMRF = 0.5; % weight for MRF constraint
% opts.active_range = [444];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im_fullres = imread(name_img);

texture_fullres = imread(name_texture);
net.normalization.averageImage = single(ones(size(im_fullres)) * opts.average);
net.normalization.imageSize = size(im_fullres);
opts.normalize = get_cnn_normalize(net.normalization) ;
opts.denormalize = get_cnn_denormalize(net.normalization) ;
opts.imgSize = net.normalization.imageSize;
net.layers = net.layers(1, 1:opts.num_layers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dream
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_ini = im_fullres;
texture_ini = texture_fullres;

% optional: only keep the low frequency info
x_ini = imresize(imresize(x_ini, 1/opts.up_scale, 'bicubic'), [size(x_ini, 1), size(x_ini, 2)], 'bicubic');

x_ini = opts.normalize(x_ini);
texture_ini = opts.normalize(texture_ini);

for i_oct = 1:opts.octave_num
    
    if i_oct  == 1
        x = imresize(x_ini, 1/opts.octave_scale^(opts.octave_num - i_oct));
        texture = imresize(texture_ini, 1/opts.octave_scale^(opts.octave_num - i_oct));
    else
        x = imresize(x, round(size(x_ini(:, :, 1)) / opts.octave_scale^(opts.octave_num - i_oct)));
        texture = imresize(texture_ini, round(size(texture_ini(:, :, 1)) / opts.octave_scale^(opts.octave_num - i_oct)));
    end
    
    if opts.lambdaMRF > 0
        coord_mrf_source = [];
        [coord_mrf_source(:, :, 1), coord_mrf_source(:, :, 2)] = meshgrid([1:opts.stride_mrf_source:size(texture, 2) - opts.patch_mrf_size, size(texture, 2) - opts.patch_mrf_size + 1], ...
            [1:opts.stride_mrf_source:size(texture, 1) - opts.patch_mrf_size, size(texture, 1) - opts.patch_mrf_size + 1]);
        coord_mrf_source = reshape(coord_mrf_source, [], 2);
        patch_mrf_source = zeros(size(coord_mrf_source, 1), opts.patch_mrf_size * opts.patch_mrf_size * opts.channel);
        for i = 1:size(coord_mrf_source, 1)
            patch = texture(coord_mrf_source(i, 2):coord_mrf_source(i, 2) + opts.patch_mrf_size - 1, coord_mrf_source(i, 1):coord_mrf_source(i, 1) + opts.patch_mrf_size - 1, :);
            patch_mrf_source(i, :) = patch(:)';
        end
    end
    
    for i_iter = 1:opts.num_iter
        i_iter
        
        res = vl_simplenn(net, x);        
        res(end).dzdx = res(end).x;

        res = vl_simplenn(net, x, res(end).dzdx);
        
        mag_mean = mean(mean(mean(mean(abs(res(1).dzdx))))) + eps(1);
        
        dr = zeros(size(x),'single'); % The MRF regularizer
        if opts.lambdaMRF > 0
            coord_mrf_target = [];
            [coord_mrf_target(:, :, 1), coord_mrf_target(:, :, 2)] = meshgrid([1:opts.stride_mrf_target:size(x, 2) - opts.patch_mrf_size, size(x, 2) - opts.patch_mrf_size + 1], ...
                [1:opts.stride_mrf_target:size(x, 1) - opts.patch_mrf_size, size(x, 1) - opts.patch_mrf_size + 1]);
            coord_mrf_target = reshape(coord_mrf_target, [], 2);
            patch_mrf_target = zeros(size(coord_mrf_target, 1), opts.patch_mrf_size * opts.patch_mrf_size * 3);
            for i = 1:size(coord_mrf_target, 1)
                patch = x(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + opts.patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + opts.patch_mrf_size - 1, :);
                patch_mrf_target(i, :) = patch(:)';
            end
            
            match_list = zeros(size(patch_mrf_target, 1), 1);
            for i = 1:size(patch_mrf_target, 1)
                diff = sum(abs(patch_mrf_source - repmat(patch_mrf_target(i, :), size(patch_mrf_source, 1), 1)), 2);
                [min_diff, match_list(i)] = min(diff);
            end
            
            mrf = 0 * x;
            count = 0 * mrf;
            for i = 1:size(patch_mrf_target, 1)
                mrf(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + opts.patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + opts.patch_mrf_size - 1, :) = ...
                    mrf(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + opts.patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + opts.patch_mrf_size - 1, :) + ...
                    reshape(patch_mrf_source(match_list(i), :), opts.patch_mrf_size, opts.patch_mrf_size, []);
                count(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + opts.patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + opts.patch_mrf_size - 1, :) = ...
                    count(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + opts.patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + opts.patch_mrf_size - 1, :) + 1;
            end
            mrf = mrf./count;
            
            % compute dr as mrf - x
            dr = opts.lambdaMRF * (mrf - x) ;
        end
        
        x = x + (opts.lr * res(1).dzdx) / mag_mean + opts.lr * dr;
        
        x(x < -opts.bound) = -opts.bound;
        x(x > opts.bound) = opts.bound;
        
        im_cnn = x + opts.average;
        imwrite(im_cnn/255, [name_output 'output_oct_' num2str(i_oct) '_' num2str(i_iter) format]);
    end
    
end

return;