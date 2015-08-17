% synthesis with gradient descent and mrf
% written by Chuan Li
% 2015/08/13
% based on matconvnet from A. Vedaldi and K. Lenc
% close all; clear all; clc;
%
% paths
path_matconvnet = '../third-party/matconvnet-dag/';
path_flann = '../third-party/flann/';
path_net = '../data/nets/';
path_input =  '../data/input/';
path_input_img_ini =  [path_input 'img_ini/'];
path_input_example_mrf =  [path_input 'example_mrf/'];
path_input_example_gd =  [path_input 'example_gd/'];
path_output = '../data/output/synthesis_matconvnet_dag_mrfgd_flann/';
mkdir(path_output);
format_output = '.png';
run ([path_matconvnet 'matlab/vl_setupnn']) ;
addpath('./Misc/deep-goggle');
addpath('./Misc/matconvet_plugin');
addpath(genpath(path_flann));

% input
name_input_net = 'caffe_bvlc_googlenet_model-dag.mat';
name_input_img_ini = 'bath_full.jpg';
name_input_example_mrf = [{'bath_full.jpg'}];
name_input_example_gd = 'baroque.jpg';

% settings
myset.matconvnet.mode = 'gpu';
myset.flann.search_params = struct('algorithm','kdtree', 'trees', 8,'checks',64, 'cores', 10);

myset.net.end_var = 'pool4_3x3_s2';
myset.net.num_var_mrf = 2;
myset.net.img_average = ones(1, 1, 3); % image mean RGB
myset.net.img_average(1) = 122;
myset.net.img_average(2) = 116;
myset.net.img_average(3) = 104;
myset.net.octave_num = 3;
myset.net.octave_scale = 2;
myset.net.octave_num_iter = 5;

myset.gd.lr = 2.5; % learning rate for gradient decsent

myset.mrf.lr = 2.5; % learning rate for mrf
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
% build net_mrf, value_example, and dict_example
% ----------------------------------------------------------------
% only keep vars that change sizes
net_mrf = [1; 1]; % [var_id; stride] in net
for i_var = 2:size(net.vars, 2)
    if strcmp(class(net.layers(i_var - 1).block), 'dagnn.Conv') || strcmp(class(net.layers(i_var - 1).block), 'dagnn.Pooling')
        if net.layers(i_var - 1).block.stride(1) > 1
            net_mrf = [net_mrf [i_var; net.layers(i_var - 1).block.stride(1)]];
        end
    end
end
net_mrf = net_mrf(:, 1:myset.net.num_var_mrf);

im_example_mrf = cell(1, size([name_input_example_mrf], 2));
for i_example = 1:size([name_input_example_mrf], 2)
    im_example_mrf{1, i_example} = single(imread([path_input_example_mrf name_input_example_mrf{1, i_example}]));
    im_example_mrf{1, i_example}(:, :, 1) = im_example_mrf{1, i_example}(:, :, 1) - myset.net.img_average(1);
    im_example_mrf{1, i_example}(:, :, 2) = im_example_mrf{1, i_example}(:, :, 2) - myset.net.img_average(2);
    im_example_mrf{1, i_example}(:, :, 3) = im_example_mrf{1, i_example}(:, :, 3) - myset.net.img_average(3);
end

% forward pass example images and stores the results
value_example_mrf = cell(1, size(name_input_example_mrf, 2));
for i_example = 1:size(name_input_example_mrf, 2)
    if strcmp(myset.matconvnet.mode, 'gpu')
        net.move('gpu');
        net.eval({'data', im_example_mrf{1, i_example}});
        net.movewithdata('cpu');
    else
        net.eval({'data', im_example_mrf{1, i_example}});
    end
    value_example_mrf{1, i_example} = net.vars;
    value_example_mrf{1, i_example} = value_example_mrf{1, i_example}(net_mrf(1, :));
    net = net_clear(net);
end

dict_example_mrf = cell(1, size(net_mrf, 2));
for i_var = 1:size(dict_example_mrf, 2)
    dict_example_mrf{1, i_var}.img_id = [];
    dict_example_mrf{1, i_var}.top_mrf = [];
    dict_example_mrf{1, i_var}.top_loc = [];
    dict_example_mrf{1, i_var}.bottom_loc = [];
end

for i_example = 1:size(name_input_example_mrf, 2)
    for i_var = 1:size(dict_example_mrf, 2)
        sz = size(value_example_mrf{1, i_example}(i_var).value);
        
        grid = [];
        [grid(:, :, 1), grid(:, :, 2)] = meshgrid([1:myset.mrf.example_stride:sz(2) - (myset.mrf.patch_size - 1) - 1, sz(2) - (myset.mrf.patch_size - 1)], ...
            [1:myset.mrf.example_stride:sz(1) - (myset.mrf.patch_size - 1) - 1, sz(1) - (myset.mrf.patch_size - 1)]);
        grid = reshape(grid, [], 2); % [[x0_cen, y0_cen]; [x1_cen, y1_cen]]
        
        dict_example_mrf{1, i_var}.img_id = [dict_example_mrf{1, i_var}.img_id; i_example * ones(size(grid, 1), 1)];
        
        % collect top_loc
        top_loc = [grid grid];
        top_loc(:, 1) = top_loc(:, 1);
        top_loc(:, 2) = top_loc(:, 2);
        top_loc(:, 3) = top_loc(:, 3) + (myset.mrf.patch_size - 1);
        top_loc(:, 4) = top_loc(:, 4) + (myset.mrf.patch_size - 1);
        dict_example_mrf{1, i_var}.top_loc = [dict_example_mrf{1, i_var}.top_loc; top_loc];
        
        % collect top_mrf
        mrf = zeros(size(grid, 1), myset.mrf.patch_size * myset.mrf.patch_size * sz(3));
        for i_patch = 1:size(mrf, 1)
            patch = value_example_mrf{1, i_example}(i_var).value(top_loc(i_patch, 2):top_loc(i_patch, 4), top_loc(i_patch, 1):top_loc(i_patch, 3), :);
            mrf(i_patch, :) = patch(:)';
        end
        dict_example_mrf{1, i_var}.top_mrf = [dict_example_mrf{1, i_var}.top_mrf; mrf];
        
        % collect bottom_loc
        if i_var > 1
            bottom_loc = net_mrf(2, i_var) * (top_loc(:, 1:2) - 1) + 1;
            bottom_loc = [bottom_loc bottom_loc + myset.mrf.patch_size * net_mrf(2, i_var) - 1];
            dict_example_mrf{1, i_var}.bottom_loc = [dict_example_mrf{1, i_var}.bottom_loc; bottom_loc];
        end
    end
end

for i_var = 1:size(dict_example_mrf, 2)
    output_flann_index = flann_build_index(dict_example_mrf{1, i_var}.top_mrf', myset.flann.search_params);
    flann_save_index(output_flann_index, [path_output 'flann_index_' num2str(i_var) '.mat']);
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
    if i_oct > 1
        x = imresize(x, [size(octaves_img{1, myset.net.octave_num - i_oct + 1}, 1), size(octaves_img{1, myset.net.octave_num - i_oct + 1}, 2)], 'bilinear');
    else
        x = octaves_img{1, myset.net.octave_num - i_oct + 1};
    end
    
    for i_iter = 1:myset.net.octave_num_iter
        disp(['oct: ' num2str(i_oct) ', i_iter: ' num2str(i_iter)]);
        
        % -------------------------------------------
        % gd
        % -------------------------------------------
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
        t_gd = t_gd + toc;
        
        % -------------------------------------------
        % mrf
        % -------------------------------------------
        tic;
        value_synthesis_mrf = net.vars;
        value_synthesis_mrf = value_synthesis_mrf(net_mrf(1, :));
        for i_var = size(value_synthesis_mrf, 2) - 1:-1:1
            value_synthesis_mrf(i_var).value = value_synthesis_mrf(i_var).value * 0;
        end
        % clear data
        net = net_clear(net);
        
        for i_var = size(value_synthesis_mrf, 2):-1:2
            % compute patches
            top_mrf = [];
            top_loc = [];
            bottom_loc = [];
            bottom_synthesis = [];
            sz = size(value_synthesis_mrf(i_var).value);
            scaler = net_mrf(2, i_var);
            sz_input = size(value_synthesis_mrf(i_var - 1).value);
            patch_size_input = myset.mrf.patch_size * scaler;
            stride_input = myset.mrf.synthesis_stride * scaler;
            grid_input = [];
            [grid_input(:, :, 1), grid_input(:, :, 2)] = meshgrid([1:stride_input:sz_input(2) - (patch_size_input - 1) - 1, sz_input(2) - (patch_size_input - 1)], ...
                [1:stride_input:sz_input(1) - (patch_size_input - 1) - 1, sz_input(1) - (patch_size_input - 1)]);
            
            grid_input = reshape(grid_input, [], 2);
            bottom_loc = [grid_input grid_input];
            bottom_loc(:, 1) = bottom_loc(:, 1);
            bottom_loc(:, 2) = bottom_loc(:, 2);
            bottom_loc(:, 3) = bottom_loc(:, 3) + (patch_size_input - 1);
            bottom_loc(:, 4) = bottom_loc(:, 4) + (patch_size_input - 1);
            
            top_loc = floor((bottom_loc(:, 1:2) - 1) / scaler) + 1;
            top_loc = [top_loc top_loc + (myset.mrf.patch_size - 1)];
            for i_patch = 1:size(top_loc, 1)
                if top_loc(i_patch, 4) > size(value_synthesis_mrf(i_var).value, 1)
                    offset = size(value_synthesis_mrf(i_var).value, 1) - top_loc(i_patch, 4);
                    top_loc(i_patch, [2, 4]) = top_loc(i_patch, [2, 4]) + offset;
                end
                if top_loc(i_patch, 3) > size(value_synthesis_mrf(i_var).value, 2)
                    offset = size(value_synthesis_mrf(i_var).value, 2) - top_loc(i_patch, 3);
                    top_loc(i_patch, [1, 3]) = top_loc(i_patch, [1, 3]) + offset;
                end
            end
            
            top_mrf = zeros(size(top_loc, 1), myset.mrf.patch_size * myset.mrf.patch_size * sz(3));
            for i_patch = 1:size(top_mrf, 1)
                patch = value_synthesis_mrf(i_var).value(top_loc(i_patch, 2):top_loc(i_patch, 4), top_loc(i_patch, 1):top_loc(i_patch, 3), :);
                top_mrf(i_patch, :) = patch(:)';
            end
            
            
            dict_example_mrf{1, i_var}.top_mrf_flann_index = flann_load_index([path_output 'flann_index_' num2str(i_var) '.mat'], dict_example_mrf{1, i_var}.top_mrf');
            
            [list_match, ndists] = flann_search(dict_example_mrf{1, i_var}.top_mrf_flann_index, top_mrf', 1, myset.flann.search_params);
            
            % synthesis
            count = 0 * value_synthesis_mrf(i_var - 1).value(:, :, 1);
            for i_patch = 1:size(list_match, 2)
                value_synthesis_mrf(i_var - 1).value(bottom_loc(i_patch, 2):bottom_loc(i_patch, 4), bottom_loc(i_patch, 1):bottom_loc(i_patch, 3), :) = value_synthesis_mrf(i_var - 1).value(bottom_loc(i_patch, 2):bottom_loc(i_patch, 4), bottom_loc(i_patch, 1):bottom_loc(i_patch, 3), :) ...
                    + value_example_mrf{1, dict_example_mrf{1, i_var}.img_id(list_match(1, i_patch))}(i_var - 1).value(dict_example_mrf{1, i_var}.bottom_loc(list_match(1, i_patch), 2):dict_example_mrf{1, i_var}.bottom_loc(list_match(1, i_patch), 4), dict_example_mrf{1, i_var}.bottom_loc(list_match(1, i_patch), 1):dict_example_mrf{1, i_var}.bottom_loc(list_match(1, i_patch), 3), :);
                count(bottom_loc(i_patch, 2):bottom_loc(i_patch, 4), bottom_loc(i_patch, 1):bottom_loc(i_patch, 3), :) = count(bottom_loc(i_patch, 2):bottom_loc(i_patch, 4), bottom_loc(i_patch, 1):bottom_loc(i_patch, 3)) + 1;
            end
            for i_depth = 1:size(value_synthesis_mrf(i_var - 1).value, 3)
                value_synthesis_mrf(i_var - 1).value(:, :, i_depth) = value_synthesis_mrf(i_var - 1).value(:, :, i_depth) ./ count;
            end
        end % for i_var = size(value_synthesis, 2):-1:size(value_synthesis, 2)
        
        
        der_mrf = value_synthesis_mrf(1).value - x;
        mag_mean_mrf = mean(mean(mean(mean(abs(der_mrf))))) + eps(1);
        
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
        t_mrf = t_mrf + toc;
        
    end
    
end

disp(['t_gd: ' num2str(t_gd) ', t_mrf: ' num2str(t_mrf)]);

