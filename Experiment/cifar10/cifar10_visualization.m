% visualize cifar10 neural network
% crucially, making sure the conv layer does NOT change the image size.
% Only the pool layer can do so. This is very important for computing the
% patch size.
% written by Chuan Li
% 2015/07/31
% based on matconvnet from A. Vedaldi and K. Lenc

close all; clear all; clc;
path_vlfeat = '/Users/chuan/Research/library/vlfeat/';
path_matconvnet = '/Users/chuan/Research/library/matconvnet/';
run ([path_vlfeat 'toolbox/vl_setup']) ;
run ([path_matconvnet 'matlab/vl_setupnn']) ;
addpath('../../Misc/deep-goggle'); 

name_model = '/Users/chuan/Research/library/matconvnet/data/cifar-lenet/net-epoch-30.mat';
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
opts.normalize = get_cnn_normalize(net.normalization) ;
opts.denormalize = get_cnn_denormalize(net.normalization) ;

% parameters
num_img = 50000;
num_candidate = 10; % keep top num_candidate for each word
bd_suppression = 3; % bandwidth for local suppression
render_margin = 4;
name_output_model = ['./net_feature_' num2str(num_img) '_top_' num2str(num_candidate) '.mat'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scale = 1; % for scaling up the patch for upper layers
for i_layer = 1:size(net.layers, 2)
    if strcmp(net.layers{1, i_layer}.type, 'conv') && net.layers{1, i_layer}.pad ~= 0
        net.layers{1, i_layer}.features = cell(1, size(net.layers{1, i_layer}.weights{1, 1}, 4));
        net.layers{1, i_layer}.features_response = cell(1, size(net.layers{1, i_layer}.weights{1, 1}, 4));
        
        for i_feature = 1:size(net.layers{1, i_layer}.features, 2)
            net.layers{1, i_layer}.features{1, i_feature} = cell(1, num_candidate);
            net.layers{1, i_layer}.features_response{1, i_feature} = zeros(1, num_candidate);
            for i_patch = 1:num_candidate
                net.layers{1, i_layer}.features{1, i_feature}{1, i_patch} = zeros(size(net.layers{1, i_layer}.weights{1, 1}, 1) * scale, ...
                    size(net.layers{1, i_layer}.weights{1, 1}, 2) * scale, ...
                    3);
            end
        end
    else
        net.layers{1, i_layer}.features = [];
        net.layers{1, i_layer}.features_response = [];
        if  strcmp(net.layers{1, i_layer}.type, 'pool')
            scale = scale * net.layers{1, i_layer}.stride;
        end
    end
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % fill in data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Progress: ');
for i_img = 1:num_img
    msg = [num2str(round(100 * i_img/num_img)) '%'];
    num2remove=numel(msg);
    disp(msg);
    
    name_img = ['/Users/chuan/Research/Dataset/data/cifar10/Train/' sprintf('%05d', i_img) '.png'];
    net.layers{end}.class = single(1) * ones(1, 1); % hack to let forward propogation work
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize source
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_ini = imread(name_img);
    im_fullres = im2double(x_ini);
    x_ini = opts.normalize(x_ini);
    res = vl_simplenn(net, x_ini);
    for i_layer = 1:size(net.layers, 2) - 1
        if ~isempty(net.layers{1, i_layer}.features)
            response = zeros(size(res(i_layer + 1).x, 1) * size(res(i_layer + 1).x, 2), 2);
            scaler = size(net.layers{1, i_layer}.features{1, 1}{1, 1}, 1) / size(net.layers{1, i_layer}.weights{1, 1}, 1);
            d = (size(net.layers{1, i_layer}.features{1, 1}{1, 1}, 1) - scaler) / 2;
            d_ = size(net.layers{1, i_layer}.features{1, 1}{1, 1}, 1);
            for i_feature = 1:size(net.layers{1, i_layer}.features, 2)
                % sort the response of output
                map_size = size(res(i_layer + 1).x(:, :, i_feature));
                response(:, 1) = reshape(res(i_layer + 1).x(:, :, i_feature), [], 1);
                response(:, 2) = [1:size(response, 1)];
                response = sortrows(response, 1);
                record_rl = [-1000; -1000];
                
                for i_candidate = 1:size(response, 1)
                    [min_response, min_idx] = min(net.layers{1, i_layer}.features_response{1, i_feature});
                    if response(end - i_candidate + 1, 1) <= min_response
                        break;
                    else
                        % find the col, row for this response on the
                        % current layer
                        [row, col] = ind2sub(map_size, response(end - i_candidate + 1, 2));
                        row_ = row; 
                        col_ = col;
                        % non-maximum suppression
                        record_rl_ = record_rl - repmat([col; row], 1, size(record_rl, 2));
                        if sum(sum(abs(record_rl_) < bd_suppression)) == 0
                            row = (row - 1) * scaler + 1;
                            col = (col - 1) * scaler + 1;
                            row_start = row - d;
                            row_end = row - d + d_ - 1;
                            col_start = col - d;
                            col_end = col - d + d_ - 1;
                            if row_start > 0 && row_end <= size(im_fullres, 1) && col_start > 0 && col_end <= size(im_fullres, 2)
                                net.layers{1, i_layer}.features{1, i_feature}{1, min_idx} = im_fullres(row_start:row_end, col_start:col_end, :);
                                net.layers{1, i_layer}.features_response{1, i_feature}(1, min_idx) = response(end - i_candidate + 1, 1);
                                record_rl = [record_rl [row_; col_]];
                            else
                                ;
                            end
                        else
                            ;
                        end
                        response(end - i_candidate + 1, 1) = 0;
                    end
                end
                 
            end
        else
            
        end
    end
    
    fprintf(repmat('\b',1,num2remove + 1));
end
disp(msg);
fprintf('\n');

% sort patches by response
for i_layer = 1:size(net.layers, 2) - 1
    if ~isempty(net.layers{1, i_layer}.features)
        for i_feature = 1:size(net.layers{1, i_layer}.features, 2)
            response = [];
            response(:, 1) = net.layers{1, i_layer}.features_response{1, i_feature}';
            response(:, 2) = [1:size(response, 1)]';
            response = sortrows(response, 1);
            idx = response(end:-1:1, 2)';
            net.layers{1, i_layer}.features{1, i_feature} = net.layers{1, i_layer}.features{1, i_feature}(idx);
        end
    else
    end
end

% compute mean
for i_layer = 1:size(net.layers, 2) - 1
    if ~isempty(net.layers{1, i_layer}.features)
        num_feature = size(net.layers{1, i_layer}.features, 2);
        net.layers{1, i_layer}.feature_mean = cell(1, num_feature);
        for i_feature = 1:num_feature
            net.layers{1, i_layer}.feature_mean{1, i_feature} = 0 * net.layers{1, i_layer}.features{1, i_feature}{1, 1};
            for i_patch = 1:size(net.layers{1, i_layer}.features{1, i_feature}, 2)
                net.layers{1, i_layer}.feature_mean{1, i_feature} = net.layers{1, i_layer}.feature_mean{1, i_feature} + net.layers{1, i_layer}.features{1, i_feature}{1, i_patch};
            end
            net.layers{1, i_layer}.feature_mean{1, i_feature} = net.layers{1, i_layer}.feature_mean{1, i_feature} / size(net.layers{1, i_layer}.features{1, i_feature}, 2);
        end
    else
        net.layers{1, i_layer}.feature_mean = [];
    end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % render and save
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i_layer = 1:size(net.layers, 2) - 1
    if ~isempty(net.layers{1, i_layer}.features)
        num_feature = size(net.layers{1, i_layer}.features, 2);
        % make full size
        h = size(net.layers{1, i_layer}.features{1, 1}{1, 1}, 1);
        w = size(net.layers{1, i_layer}.features{1, 1}{1, 1}, 2);
        
        im_out = ones(h * num_feature + render_margin * (num_feature - 1), w * (num_candidate + 1) + render_margin * (num_candidate), 3);
        for i_feature = 1:size(net.layers{1, i_layer}.features, 2)
            im_out((i_feature - 1) * (h + render_margin) + 1:(i_feature - 1) * (h + render_margin) + h, 1:w, :) = net.layers{1, i_layer}.feature_mean{1, i_feature};
            for i_patch = 1:size(net.layers{1, i_layer}.features{1, i_feature}, 2)
                im_out((i_feature - 1) * (h + render_margin) + 1:(i_feature - 1) * (h + render_margin) + h, (i_patch) * (w + render_margin) + 1:(i_patch) * (w + render_margin) + w, :) = net.layers{1, i_layer}.features{1, i_feature}{1, i_patch};
            end
        end
        im_out = im_out;
        imwrite(im_out, ['visual_layer_' num2str(i_layer) '_top_' num2str(num_candidate) '_img_' num2str(num_img) '.png']);
    else
    end
end

save(name_output_model, 'net');


return;





















