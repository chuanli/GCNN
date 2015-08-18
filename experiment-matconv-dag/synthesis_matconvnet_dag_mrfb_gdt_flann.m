% synthesis with mrf bottom and gradient descent top
% written by Chuan Li
% 2015/08/17
% based on matconvnet from A. Vedaldi and K. Lenc
close all; clear all; clc;

% paths
path_matconvnet = '../third-party/matconvnet-dag/';
path_flann = '../third-party/flann/';
path_net = '../data/nets/';
path_input =  '../data/input/';
path_input_img_ini =  [path_input 'img_ini/'];
path_input_example_mrf =  [path_input 'example_mrf/'];
path_input_example_gd =  [path_input 'example_gd/'];
path_output = '../data/output/synthesis_matconvnet_dag_mrfb_gdt_flann/';
mkdir(path_output);
format_output = '.png';
run ([path_matconvnet 'matlab/vl_setupnn']) ;
addpath('./Misc/deep-goggle');
addpath('./Misc/matconvet_plugin');
addpath(genpath(path_flann));

% input
name_input_net = 'caffe_bvlc_googlenet_model-dag.mat';
name_input_img_ini = 'bath_full.jpg';
name_input_example_mrf = [{'baroque.jpg'}];
name_input_example_gd = 'baroque.jpg';

% settings
myset.matconvnet.mode = 'cpu';
myset.flann.search_params = struct('algorithm','kdtree', 'trees', 8,'checks',64, 'cores', 10);

myset.net.end_var = 'pool4_3x3_s2';
myset.net.num_var_mrf = 2;
myset.net.img_average = ones(1, 1, 3); % image mean RGB
myset.net.img_average(1) = 122;
myset.net.img_average(2) = 116;
myset.net.img_average(3) = 104;
myset.net.octave_num = 3;
myset.net.octave_scale = 2;
myset.net.octave_num_iter = 10;

myset.gd.lr = 1.5; % learning rate for gradient decsent

myset.mrf.lr = 0.5; % learning rate for mrf
myset.mrf.patch_size = 3; % mrf patch size
myset.mrf.example_stride = 5; % mrf example stride
myset.mrf.synthesis_stride = 1; % mrf synthesis stride

load([path_net name_input_net]) ;

net = dagnn.DagNN.loadobj(netDAG) ;
myset.net.num_vars = net.getVarIndex(myset.net.end_var);
for i = size(net.vars, 2):-1:myset.net.num_vars + 1
    net.removeLayer(net.layers(end).name);
end
net.conserveMemory = 0;

% ----------------------------------------------------------------
% create examples for gradient descenet (only for the top var)
% ----------------------------------------------------------------
example_gd_fullres = imread([path_input_example_gd name_input_example_gd]);
example_gd_ini = single(example_gd_fullres);
example_gd_ini(:, :, 1) = example_gd_ini(:, :, 1) - myset.net.img_average(1);
example_gd_ini(:, :, 2) = example_gd_ini(:, :, 2) - myset.net.img_average(2);
example_gd_ini(:, :, 3) = example_gd_ini(:, :, 3) - myset.net.img_average(3);

if strcmp(myset.matconvnet.mode, 'gpu')
    net.move('gpu');
    net.eval({'data', example_gd_ini});
    net.movewithdata('cpu');
else
    net.eval({'data', example_gd_ini});
end

var_example_top_gd = net.vars(end).value;
patch_example_top_gd = zeros(size(var_example_top_gd, 1) * size(var_example_top_gd, 2), size(var_example_top_gd, 3));
for i = 1:size(patch_example_top_gd, 1)
    [row, col] = ind2sub(size(var_example_top_gd(:, :, 1)), i);
    patch_example_top_gd(i, :) = reshape(var_example_top_gd(row, col, :), 1, []);
end

net = net_clear(net);

% ----------------------------------------------------------------
% read in example mrf images
% ----------------------------------------------------------------
example_mrf_fullres = cell(1, size([name_input_example_mrf], 2));
for i_example = 1:size([name_input_example_mrf], 2)
    example_mrf_fullres{1, i_example} = single(imread([path_input_example_mrf name_input_example_mrf{1, i_example}]));
    example_mrf_fullres{1, i_example}(:, :, 1) = example_mrf_fullres{1, i_example}(:, :, 1) - myset.net.img_average(1);
    example_mrf_fullres{1, i_example}(:, :, 2) = example_mrf_fullres{1, i_example}(:, :, 2) - myset.net.img_average(2);
    example_mrf_fullres{1, i_example}(:, :, 3) = example_mrf_fullres{1, i_example}(:, :, 3) - myset.net.img_average(3);
end

% ----------------------------------------------------------------
% synthesis
% ----------------------------------------------------------------
im_synthesis = imread([path_input_img_ini name_input_img_ini]);
im_synthesis = single(im_synthesis);
im_synthesis(:, :, 1) = im_synthesis(:, :, 1) - myset.net.img_average(1);
im_synthesis(:, :, 2) = im_synthesis(:, :, 2) - myset.net.img_average(2);
im_synthesis(:, :, 3) = im_synthesis(:, :, 3) - myset.net.img_average(3);

octaves_img = cell(1, myset.net.octave_num);
for i_oct = 1:myset.net.octave_num
    if i_oct > 1
        octaves_img{1, i_oct} = imresize(octaves_img{1, 1}, 1/(myset.net.octave_scale ^ (i_oct - 1)), 'bilinear');
    else
        octaves_img{1, i_oct} = im_synthesis;
    end
end


t_gd = 0;
t_mrf = 0;
for i_oct = 1:myset.net.octave_num
% for i_oct = 1:1
    if i_oct > 1
        x = imresize(x, [size(octaves_img{1, myset.net.octave_num - i_oct + 1}, 1), size(octaves_img{1, myset.net.octave_num - i_oct + 1}, 2)], 'bilinear');
    else
        x = octaves_img{1, myset.net.octave_num - i_oct + 1};
    end
    
    if myset.mrf.lr > 0
        patch_example_bottom_mrf = [];
        example_mrf = cell(1, size([name_input_example_mrf], 2));
        for i_example = 1:size([name_input_example_mrf], 2)
            example_mrf{1, i_example} = imresize(example_mrf_fullres{1, i_example}, 1/myset.net.octave_scale^(myset.net.octave_num - i_oct)) ;
            coord_mrf = [];
            [coord_mrf(:, :, 1), coord_mrf(:, :, 2)] = meshgrid([1:myset.mrf.example_stride:size(example_mrf{1, i_example}, 2) - myset.mrf.patch_size + 1], ...
                [1:myset.mrf.example_stride:size(example_mrf{1, i_example}, 1) - myset.mrf.patch_size + 1]);
            coord_mrf = reshape(coord_mrf, [], 2);
            patch_mrf = zeros(size(coord_mrf, 1), myset.mrf.patch_size * myset.mrf.patch_size * 3);
            for i = 1:size(coord_mrf, 1)
                patch = example_mrf{1, i_example}(coord_mrf(i, 2):coord_mrf(i, 2) + myset.mrf.patch_size - 1, coord_mrf(i, 1):coord_mrf(i, 1) + myset.mrf.patch_size - 1, :);
                patch_mrf(i, :) = patch(:)';
            end
            patch_example_bottom_mrf = [patch_example_bottom_mrf; patch_mrf];
        end
        
        % build flann index
        output_flann_index = flann_build_index(patch_example_bottom_mrf', myset.flann.search_params);
        flann_save_index(output_flann_index, [path_output 'flann_index_' num2str(i_oct) '.mat']);
    end
    
    for i_iter = 1:myset.net.octave_num_iter
%     for i_iter = 1:1
        disp(['oct: ' num2str(i_oct) ', i_iter: ' num2str(i_iter)]);
        
        % -------------------------------------------
        % gd
        % -------------------------------------------
        der_gd = 0 * x;
        mag_mean_gd = 1;
        
        tic;
        if strcmp(myset.matconvnet.mode, 'gpu')
            net.move('gpu');
            net.eval({'data', x});
            net.movewithdata('cpu');
        else
            net.eval({'data', x});
        end
        var_synthesis_top_gd = net.vars(end).value;
        
        patch_synthesis_top_gd = zeros(size(var_synthesis_top_gd, 1) * size(var_synthesis_top_gd, 2), size(var_synthesis_top_gd, 3));
        for i = 1:size(patch_synthesis_top_gd, 1)
            [row, col] = ind2sub(size(var_synthesis_top_gd(:, :, 1)), i);
            patch_synthesis_top_gd(i, :) = reshape(var_synthesis_top_gd(row, col, :), 1, []);
        end
        
        patch_match = 0 * patch_synthesis_top_gd;
        list_match = 0 * ones(2, size(patch_match, 1));
        for i_patch = 1:size(patch_match, 1)
            dotp = sum(patch_example_top_gd .* repmat(patch_synthesis_top_gd(i_patch, :), size(patch_example_top_gd, 1), 1), 2);
            [max_dotp, max_id] = max(dotp);
            patch_match(i_patch, :) = patch_example_top_gd(max_id, :);
            list_match(:, i_patch) = [max_id; max_dotp];
        end
        
        der_top_gd = var_synthesis_top_gd * 0;
        for i = 1:size(patch_match, 1)
            [row, col] = ind2sub(size(var_synthesis_top_gd(:, :, 1)), i);
            der_top_gd(row, col, :) = reshape(patch_match(i, :), 1, 1, []);
        end
        
        der_top_gd = der_top_gd - var_synthesis_top_gd;
        
        if strcmp(myset.matconvnet.mode, 'gpu')
            net.movewithdata('gpu');
            net.eval_backprop({'data', x}, {myset.net.end_var, gpuArray(der_top_gd)}, myset.net.end_var);
            net.movewithdata('cpu');
        else
            net.eval_backprop({'data', x}, {myset.net.end_var, der_top_gd}, myset.net.end_var);
        end
        
        der_gd = gather(net.vars(1).der);
        mag_mean_gd = mean(mean(mean(mean(abs(der_gd))))) + eps(1);
        net = net_clear(net);
        t_gd = t_gd + toc;
        
        % -------------------------------------------
        % mrf
        % -------------------------------------------
        der_mrf = 0 * x;
        mag_mean_mrf = 1;
        
        tic;
        if myset.mrf.lr > 0
            patch_synthesis_bottom_mrf = [];
            coord_synthesis_bottom_mrf = [];
            [coord_synthesis_bottom_mrf(:, :, 1), coord_synthesis_bottom_mrf(:, :, 2)] = meshgrid([1:myset.mrf.synthesis_stride:size(x, 2) - myset.mrf.patch_size, size(x, 2) - myset.mrf.patch_size + 1], ...
                [1:myset.mrf.synthesis_stride:size(x, 1) - myset.mrf.patch_size, size(x, 1) - myset.mrf.patch_size + 1]);
            coord_synthesis_bottom_mrf = reshape(coord_synthesis_bottom_mrf, [], 2);
            patch_synthesis_bottom_mrf = zeros(size(coord_synthesis_bottom_mrf, 1), myset.mrf.patch_size * myset.mrf.patch_size * 3);
            for i = 1:size(coord_synthesis_bottom_mrf, 1)
                patch = x(coord_synthesis_bottom_mrf(i, 2):coord_synthesis_bottom_mrf(i, 2) + myset.mrf.patch_size - 1, coord_synthesis_bottom_mrf(i, 1):coord_synthesis_bottom_mrf(i, 1) + myset.mrf.patch_size - 1, :);
                patch_synthesis_bottom_mrf(i, :) = patch(:)';
            end
            
            flann_index = flann_load_index([path_output 'flann_index_' num2str(i_oct) '.mat'], patch_example_bottom_mrf');
            [match_list, ndists] = flann_search(flann_index, patch_synthesis_bottom_mrf', 1, myset.flann.search_params);
            
%             match_list = zeros(size(patch_synthesis_bottom_mrf, 1), 1);
%             for i = 1:size(patch_synthesis_bottom_mrf, 1)
%                 diff = sum(abs(patch_example_bottom_mrf - repmat(patch_synthesis_bottom_mrf(i, :), size(patch_example_bottom_mrf, 1), 1)), 2);
%                 [min_diff, match_list(i)] = min(diff);
%             end
            
            mrf = 0 * x;
            count = 0 * mrf(:, :, 1);
            for i = 1:size(patch_synthesis_bottom_mrf, 1)
                mrf(coord_synthesis_bottom_mrf(i, 2):coord_synthesis_bottom_mrf(i, 2) + myset.mrf.patch_size - 1, coord_synthesis_bottom_mrf(i, 1):coord_synthesis_bottom_mrf(i, 1) + myset.mrf.patch_size - 1, :) = ...
                    mrf(coord_synthesis_bottom_mrf(i, 2):coord_synthesis_bottom_mrf(i, 2) + myset.mrf.patch_size - 1, coord_synthesis_bottom_mrf(i, 1):coord_synthesis_bottom_mrf(i, 1) + myset.mrf.patch_size - 1, :) + ...
                    reshape(patch_example_bottom_mrf(match_list(i), :), myset.mrf.patch_size, myset.mrf.patch_size, []);
                count(coord_synthesis_bottom_mrf(i, 2):coord_synthesis_bottom_mrf(i, 2) + myset.mrf.patch_size - 1, coord_synthesis_bottom_mrf(i, 1):coord_synthesis_bottom_mrf(i, 1) + myset.mrf.patch_size - 1, :) = ...
                    count(coord_synthesis_bottom_mrf(i, 2):coord_synthesis_bottom_mrf(i, 2) + myset.mrf.patch_size - 1, coord_synthesis_bottom_mrf(i, 1):coord_synthesis_bottom_mrf(i, 1) + myset.mrf.patch_size - 1, :) + 1;
            end
            mrf(:, :, 1) = mrf(:, :, 1)./count;
            mrf(:, :, 2) = mrf(:, :, 2)./count;
            mrf(:, :, 3) = mrf(:, :, 3)./count;
            
            % compute dr as mrf - x
            der_mrf = mrf - x ;
        end
        t_mrf = t_mrf + toc;
        
        x = x + (myset.gd.lr * der_gd)  / mag_mean_gd + (myset.mrf.lr * der_mrf)  / mag_mean_mrf;
        x_red = x(:, :, 1);
        x_green = x(:, :, 2);
        x_blue = x(:, :, 3);
        x_red(x_red < -myset.net.img_average(1)) = -myset.net.img_average(1);
        x_red(x_red > 255-myset.net.img_average(1)) = 255-myset.net.img_average(1);
        x_green(x_green < -myset.net.img_average(2)) = -myset.net.img_average(2);
        x_green(x_green > 255-myset.net.img_average(2)) = 255-myset.net.img_average(2);
        x_blue(x_blue < -myset.net.img_average(3)) = -myset.net.img_average(3);
        x_blue(x_blue > 255-myset.net.img_average(3)) = 255-myset.net.img_average(3);
        x = cat(3, x_red, x_green, x_blue);
        
        im_cnn = x;
        im_cnn(:, :, 1) = im_cnn(:, :, 1) + myset.net.img_average(1);
        im_cnn(:, :, 2) = im_cnn(:, :, 2) + myset.net.img_average(2);
        im_cnn(:, :, 3) = im_cnn(:, :, 3) + myset.net.img_average(3);
        imwrite(im_cnn/255, [path_output 'output_oct_' num2str(i_oct) '_' num2str(i_iter) format_output]);        
    end
    
    
end


disp(['t_gd: ' num2str(t_gd) ', t_mrf: ' num2str(t_mrf)]);



