% one hot generation with mrf prior
% written by Chuan Li
% 2015/07/31
% based on matconvnet from A. Vedaldi and K. Lenc
close all; clear all; clc;
path_vlfeat = '/Users/chuan/Research/library/vlfeat/';
path_matconvnet = '/Users/chuan/Research/library/matconvnet/';
run ([path_vlfeat 'toolbox/vl_setup']) ;
run ([path_matconvnet 'matlab/vl_setupnn']) ;
addpath('../../Misc/deep-goggle'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup experiments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
name_model = '/Users/chuan/Research/library/matconvnet/data/cifar-lenet/net-epoch-30.mat';
name_deepgoggle_onehot_mrf = '/Users/chuan/Research/Dataset/data/cifar10/deepgoggle_onehot_mrf/';
format = '.png';
if(~exist(name_model, 'file'))
    warning('Pretrained model not found\n');
end
load(name_model);
if ~isfield(net, 'normalization')
    name_imdb = '/Users/chuan/Research/library/matconvnet/data/cifar-lenet/imdb.mat';
    imdb = load(name_imdb);
    net.normalization.averageImage = imdb.averageImage;
    net.normalization.imageSize = imdb.imageSize;
    net.class = imdb.meta.classes;
    net.label = imdb.images.labels;
    save('/Users/chuan/Research/library/matconvnet/data/cifar-lenet/net-epoch-30.mat', 'net');
end

% option
opts.learningRate = 0.002 * [...
    ones(1,100)];
opts.lambdaMRF = 200; % weight for MRF constraint
opts.momentum = 0.0 ; % Momentum used in the optimization
opts.normalize = get_cnn_normalize(net.normalization) ;
opts.denormalize = get_cnn_denormalize(net.normalization) ;
opts.imgSize = net.normalization.imageSize;

% parameters
target_class = 2;
up_scale = 3;
num_t = 100;

patch_mrf_size = 3;
stride_mrf_source = 1;
stride_mrf_target = 1;
dict_step_rotation = 10;
dict_num_rotation = 1;

num_img = 10;
output_step = 10;
output_padding = 5;

for i_img = 1:num_img
    name_img = ['/Users/chuan/Research/Dataset/data/cifar10/Test/' sprintf('%05d', i_img) '.png'];
    
    im_output = [];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize target
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    net.layers{end}.class = single(target_class) * ones(1, 1);
    sigma_target = 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize source
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % intialize using a real image
    im_fullres = imread(name_img);
    im_output = [im_output double(im_fullres)/255 zeros(size(im_fullres, 1), output_padding, 3)];
    
    x_ini = im_fullres;
    x_ini = imresize(imresize(x_ini, 1/up_scale, 'bicubic'), [size(x_ini, 1), size(x_ini, 2)], 'bicubic');
    x_ini = opts.normalize(x_ini);
    sigma_ini = norm(x_ini(:));
    x_momentum = zeros(opts.imgSize, 'single');
    im_ini = opts.denormalize(x_ini);
               
    load(['dict_class_' num2str(target_class) '_size_' num2str(patch_mrf_size) '.mat']);
    patch_mrf_source = dict_class;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CNN optimization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % get an initial estimation
    res = vl_simplenn(net, x_ini);
    prob_ini = vl_nnsoftmax(res(end - 1).x);
    prob_ini = prob_ini(:);
    prob_target = zeros(size(prob_ini), 'single');
    prob_target(target_class, 1) = 1;
    
    % record results
    prevlr = 0 ;
    x = x_ini;
    im_cnn = vl_imsc(opts.denormalize(x)) * 255;
    im_output = [im_output double(im_cnn)/255 zeros(size(im_fullres, 1), output_padding, 3)];
    
    for t=1:num_t
        
        % Effectively does both forward and backward passes
        res = vl_simplenn(net, x, single(1)) ;
        
        % check learning rate
        lr = opts.learningRate(min(t, numel(opts.learningRate)));
        if lr ~= prevlr
            x_momentum = 0 * x_momentum ;
            prevlr = lr;
        end
        
        dr = zeros(size(x),'single'); % The MRF regularizer
        
        if opts.lambdaMRF > 0
            % compute im_mrf, the reconstruction of im_cnn with MRF dictionary
            coord_mrf_target = [];
            [coord_mrf_target(:, :, 1), coord_mrf_target(:, :, 2)] = meshgrid([1:stride_mrf_target:size(im_cnn, 2) - patch_mrf_size, size(im_cnn, 2) - patch_mrf_size + 1], ...
                [1:stride_mrf_target:size(im_cnn, 1) - patch_mrf_size, size(im_cnn, 1) - patch_mrf_size + 1]);
            coord_mrf_target = reshape(coord_mrf_target, [], 2);
            patch_mrf_target = zeros(size(coord_mrf_target, 1), patch_mrf_size * patch_mrf_size * 3);
            for i = 1:size(coord_mrf_target, 1)
                patch = im_cnn(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + patch_mrf_size - 1, :);
                patch_mrf_target(i, :) = patch(:)';
            end
            
            match_list = zeros(size(patch_mrf_target, 1), 1);
            for i = 1:size(patch_mrf_target, 1)
                diff = sum(abs(patch_mrf_source - repmat(patch_mrf_target(i, :), size(patch_mrf_source, 1), 1)), 2);
                [min_diff, match_list(i)] = min(diff);
            end
            
            im_mrf = 0 * im_cnn;
            im_count = 0 * im_cnn;
            for i = 1:size(patch_mrf_target, 1)
                im_mrf(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + patch_mrf_size - 1, :) = ...
                    im_mrf(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + patch_mrf_size - 1, :) + ...
                    reshape(patch_mrf_source(match_list(i), :), patch_mrf_size, patch_mrf_size, []);
                im_count(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + patch_mrf_size - 1, :) = ...
                    im_count(coord_mrf_target(i, 2):coord_mrf_target(i, 2) + patch_mrf_size - 1, coord_mrf_target(i, 1):coord_mrf_target(i, 1) + patch_mrf_size - 1, :) + 1;
            end
            im_mrf = im_mrf./im_count;
            
            % compute dr as im_mrf - im_cnn
            dr = opts.lambdaMRF * (im_mrf - im_cnn) ;
        end
        
        
        % x_momentum combines the current gradient and the previous gradients
        % with decay (opts.momentum)
        x_momentum = opts.momentum * x_momentum ...
            + lr * dr ...
            - (lr * sigma_ini^2/sigma_target^2) * res(1).dzdx;
        
        % This is the main update step (we are updating the the variable
        % along the gradient
        x = x + x_momentum ;
        prob_cnn = vl_nnsoftmax(res(end - 1).x);
        prob_cnn = prob_cnn(:);
        im_cnn = vl_imsc(opts.denormalize(x)) * 255;
        
        %         imwrite(im_cnn/255, [name_deepgoggle_onehot_mrf sprintf('%05d', i_img) '_' num2str(t) '.png']);
        if mod(t-1,output_step)==0 || t == num_t
            im_output = [im_output double(im_cnn)/255 zeros(size(im_fullres, 1), output_padding, 3)];
        end
    end
    
    imwrite(im_output, [name_deepgoggle_onehot_mrf sprintf('%05d', i_img) '_class_' num2str(target_class) '.png']);
end