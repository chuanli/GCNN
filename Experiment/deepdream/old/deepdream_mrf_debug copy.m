% % deepdream with mrf prior
% % written by Chuan Li
% % 2015/07/31
% % based on matconvnet from A. Vedaldi and K. Lenc
close all; clear all; clc;
path_vlfeat = '/Users/chuan/Research/library/vlfeat/';
path_matconvnet = '/Users/chuan/Research/library/matconvnet/';
run ([path_vlfeat 'toolbox/vl_setup']) ;
run ([path_matconvnet 'matlab/vl_setupnn']) ;
addpath('../../Misc/deep-goggle');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup experiments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
name_model = './net_feature_50_top_10.mat';
name_googledream = '/Users/chuan/Research/Dataset/data/deepdream/';
name_output = '/Users/chuan/Research/Dataset/data/deepdream/mrf/';
format = '.png';
if(~exist(name_model, 'file'))
    warning('Pretrained model not found\n');
end
load(name_model);

name_img = ['/Users/chuan/Research/Dataset/data/deepdream/bath.jpg'];
name_texture = ['/Users/chuan/Research/Dataset/data/deepdream/bath.jpg'];

% parameters
opts.num_layers = 35;
opts.octave_num = 3;
opts.octave_scale = 2;
opts.num_iter = 20;
opts.lr =1.5; % learning rate

opts.up_scale = 8;
opts.average = 128; % image mean
opts.bound = 255 - opts.average;
opts.channel = 3;

opts.patch_mrf_size = size(net.layers{1, opts.num_layers}.weights{1, 1}, 1);
opts.stride_mrf_source = 1;
opts.stride_mrf_target = 1;

% opts.lambdaMRF = 0.5; % weight for MRF constraint
opts.lambdaMRF = 0; % weight for MRF constraint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im_fullres = imread(name_img);
im_fullres = imresize(im_fullres, [64 * 4, 64 * 3]);
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
% x_fullres = opts.normalize(im_fullres);
% res_x_fullres = vl_simplenn(net, x_fullres);
% 
% x_blur = opts.normalize(imresize(imresize(im_fullres, 1/opts.up_scale, 'bicubic'), [size(im_fullres, 1), size(im_fullres, 2)], 'bicubic'));
% res_x_blur = vl_simplenn(net, x_blur);
% 
% sel_feature = [1, 1];
% feature_fullres = reshape(res_x_fullres(end).x(sel_feature(1), sel_feature(2), :), 1, []);
% feature_blur = reshape(res_x_blur(end).x(sel_feature(1), sel_feature(2), :), 1, []);
% 
% feature_fullres_normalized = (feature_fullres - min(feature_fullres))/(max(feature_fullres) - min(feature_fullres)) - 0.5;
% feature_blur_normalized = (feature_blur - min(feature_blur))/(max(feature_blur) - min(feature_blur)) - 0.5;
% 
% close all;
% 
% figure; clf ;
% h1 = subplot(1, 3, 1) ;
% plot([1:size(feature_fullres, 2)], feature_fullres, 'r');
% axis([0 size(feature_fullres, 2) -500 500]);
% 
% h2 = subplot(1, 3, 2) ;
% plot([1:size(feature_blur, 2)], feature_blur, 'r');
% axis([0 size(feature_blur, 2) -500 500]);
% 
% h3 = subplot(1, 3, 3) ;
% hold on;
% plot([1:size(feature_fullres_normalized, 2)], feature_fullres_normalized, 'r');
% plot([1:size(feature_blur_normalized, 2)], feature_blur_normalized, 'b');
% axis([0 size(feature_fullres_normalized, 2) -1 1]);
% 
% sum(feature_fullres_normalized.*feature_blur_normalized) / (norm(feature_fullres_normalized) * norm(feature_blur_normalized))
% 
% % h3 = subplot(1, 4, 4) ;
% % plot([1:size(feature_blur_normalized, 2)], feature_blur_normalized, 'b');
% % axis([0 size(feature_blur_normalized, 2) -500 500]);
% 
% drawnow;
%             
% return;

x_ini = im_fullres;
texture_ini = texture_fullres;

% only keep the low frequency info
x_ini = imresize(imresize(x_ini, 1/opts.up_scale, 'bicubic'), [size(x_ini, 1), size(x_ini, 2)], 'bicubic');

% normalize for cnn
x_ini = opts.normalize(x_ini);
texture_ini = opts.normalize(texture_ini);

% build pyramids
octaves_img = cell(1, opts.octave_num);
for i_oct = 1:opts.octave_num
    octaves_img{1, i_oct} = imresize(x_ini, 1/opts.octave_scale^(i_oct - 1));
end

octaves_texture = cell(1, opts.octave_num);
for i_oct = 1:opts.octave_num
    octaves_texture{1, i_oct} = imresize(texture_ini, 1/opts.octave_scale^(i_oct - 1));
end

for i_oct = 1:opts.octave_num
    % for i_oct = 1:1
    x = octaves_img{1, opts.octave_num - i_oct + 1};
    texture = octaves_texture{1, opts.octave_num - i_oct + 1};
    im_cnn = x + opts.average;
    im_cnn_start = im_cnn;
    
    % building dictionary
    res_texture = vl_simplenn(net, texture);
    response_texture = res_texture(end).x;
    
    coord_mrf_source = [];
    [coord_mrf_source(:, :, 1), coord_mrf_source(:, :, 2)] = meshgrid([(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_source:size(response_texture, 2) - (opts.patch_mrf_size - 1)/2 - 1, size(response_texture, 2) - (opts.patch_mrf_size - 1)/2], ...
        [(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_source:size(response_texture, 1) - (opts.patch_mrf_size - 1)/2 - 1, size(response_texture, 1) - (opts.patch_mrf_size - 1)/2]);
    coord_mrf_source = reshape(coord_mrf_source, [], 2);
    scaler2fullres = size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1) / opts.patch_mrf_size;
    coord_mrf_source_fullres =  coord_mrf_source * scaler2fullres - scaler2fullres/2 + 1;
    
    patch_mrf_source = zeros(size(coord_mrf_source, 1), size(response_texture, 3));
    for i = 1:size(coord_mrf_source, 1)
        patch = response_texture(coord_mrf_source(i, 2), coord_mrf_source(i, 1), :);
        patch_mrf_source(i, :) = patch(:)';
    end
%     patch_mrf_source(patch_mrf_source < 0) = 0;
    patch_mrf_source_fullres = zeros(size(coord_mrf_source, 1), size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1) * size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 2) * opts.channel);
    for i = 1:size(coord_mrf_source, 1)
        patch = texture(coord_mrf_source_fullres(i, 2) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_source_fullres(i, 2) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, ...
            coord_mrf_source_fullres(i, 1) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_source_fullres(i, 1) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, :);
        patch_mrf_source_fullres(i, :) = patch(:)';
    end
    
    if i_oct > 1
        % upscale details from the previous octave
        detail = imresize(detail, [size(x, 1), size(x, 2)]);
    else
        detail = x * 0;
    end
    
    x_ini = x;
    x = x + detail;
    
    res_x_ref = vl_simplenn(net, x);
    
    for i_iter = 1:opts.num_iter
        
        res_x = vl_simplenn(net, x);
        
%         res_x(end).dzdx = res_x(end).x;
%         res_x(end).dzdx(res_x(end).dzdx < 0) = 0;
            
        res_x(end).dzdx = res_x(end).x;
%        res_x(end).dzdx(res_x_ref(end).x < 0) = res_x_ref(end).x(res_x_ref(end).x < 0) - res_x(end).x(res_x_ref(end).x < 0);
                          
        res_x = vl_simplenn(net, x, res_x(end).dzdx);
        
        mag_mean = mean(mean(mean(mean(abs(res_x(1).dzdx))))) + 0.0000000001;
        
        x = x + (opts.lr * res_x(1).dzdx) / mag_mean;
        
%         x(x < -opts.bound) = -opts.bound;
%         x(x > opts.bound) = opts.bound;
        
        im_cnn = x + opts.average;
        
        imwrite(im_cnn/255, [name_output 'output_oct_' num2str(i_oct) '_' num2str(i_iter) format]);
    end
    
    detail = x - x_ini;
            
    if i_oct == opts.octave_num 
        % do debug
        response_x = res_x(end).x;
        coord_mrf_target = [];
        [coord_mrf_target(:, :, 1), coord_mrf_target(:, :, 2)] = meshgrid([(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_target:size(response_x, 2) - (opts.patch_mrf_size - 1)/2 - 1, size(response_x, 2) - (opts.patch_mrf_size - 1)/2], ...
            [(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_target:size(response_x, 1) - (opts.patch_mrf_size - 1)/2 - 1, size(response_x, 1) - (opts.patch_mrf_size - 1)/2]);
        coord_mrf_target = reshape(coord_mrf_target, [], 2);
        coord_mrf_target_fullres =  coord_mrf_target * scaler2fullres - scaler2fullres/2 + 1;
        patch_mrf_target = zeros(size(coord_mrf_target, 1), size(response_x, 3));
        for i = 1:size(coord_mrf_target, 1)
            patch = response_x(coord_mrf_target(i, 2), coord_mrf_target(i, 1), :);
            patch_mrf_target(i, :) = patch(:)';
        end
%         patch_mrf_target(patch_mrf_target < 0) = 0;
        match_list = zeros(size(patch_mrf_target, 1), 1);
%         weight_source = max(abs(patch_mrf_source), [], 2);
%         weight_target = max(abs(patch_mrf_target), [], 2);
%         patch_mrf_source = patch_mrf_source./repmat(weight_source, 1, size(patch_mrf_source, 2));
%         patch_mrf_target = patch_mrf_target./repmat(weight_target, 1, size(patch_mrf_target, 2));
        
        for i = 1:size(patch_mrf_target, 1)
            diff = sum(abs(patch_mrf_source - repmat(patch_mrf_target(i, :), size(patch_mrf_source, 1), 1)), 2);
            [min_diff, match_list(i)] = min(diff);
%               corre = sum(patch_mrf_source.*repmat(patch_mrf_target(i, :), size(patch_mrf_source, 1), 1), 2);
%               [max_corr, match_list(i)] = max(corre);
        end
        
        close all;
        figure;
        hold on;
        for i_p = 1:size(patch_mrf_target, 1)
            plot([1:size(patch_mrf_target, 2)], patch_mrf_target(i_p, :), 'r');
        end
        axis([0 size(patch_mrf_target, 2) -1500 1500]) 
        drawnow;
%         close all;
%         for i_p = 1:10    
%             im_mrf = 0 * im_cnn;
%             im_count = 0 * im_cnn;
%             %            for i = 1:size(patch_mrf_target, 1)
%             for i = i_p:i_p
%                 im_mrf(coord_mrf_target_fullres(i, 2) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 2) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, coord_mrf_target_fullres(i, 1) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 1) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, :) = ...
%                     im_mrf(coord_mrf_target_fullres(i, 2) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 2) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, coord_mrf_target_fullres(i, 1) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 1) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, :) + ...
%                     reshape(patch_mrf_source_fullres(match_list(i), :), size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1), size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1), []);
%                 im_count(coord_mrf_target_fullres(i, 2) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 2) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, coord_mrf_target_fullres(i, 1) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 1) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, :) = ...
%                     im_count(coord_mrf_target_fullres(i, 2) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 2) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, coord_mrf_target_fullres(i, 1) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 1) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, :) + 1;
%             end
%             im_count(im_count == 0) = 1;
%             im_mrf = im_mrf./im_count;
%             im_mrf = im_mrf + opts.average;
%       
%             im_ref = 0 * im_cnn;
%             for i = i_p:i_p
%                 im_ref(coord_mrf_target_fullres(i, 2) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 2) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, coord_mrf_target_fullres(i, 1) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 1) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, :) = ...
%                     im_cnn(coord_mrf_target_fullres(i, 2) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 2) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, coord_mrf_target_fullres(i, 1) - size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2:coord_mrf_target_fullres(i, 1) + size(net.layers{1, opts.num_layers}.features{1, 1}{1, 1}, 1)/2 - 1, :);
%             end
%             
%             % display
%             figure; clf ;  
%             h1 = subplot(1, 3, 1) ;
%             plot([1:size(patch_mrf_source, 2)], patch_mrf_source(i_p, :), 'r');
%             axis([0 size(patch_mrf_source, 2) -500 500])
% 
%             h2 = subplot(1, 3, 2) ;
%             plot([1:size(patch_mrf_target, 2)], patch_mrf_target(i_p, :), 'r');
%             axis([0 size(patch_mrf_target, 2) -500 500])           
%   
%             h3 = subplot(1, 3, 3) ;
%             plot([1:size(patch_mrf_target, 2)], patch_mrf_source(match_list(i_p), :), 'r');
%             axis([0 size(patch_mrf_target, 2) -500 500]) 
%             
%             drawnow;
%             
%         end
    end
end











