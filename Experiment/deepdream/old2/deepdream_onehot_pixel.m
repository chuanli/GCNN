% deepdream with mrf prior
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
name_model = '/Users/chuan/Research/library/matconvnet/data/imagenet/imagenet-vgg-verydeep-19.mat';
name_googledream = '/Users/chuan/Research/Dataset/data/deepdream/';
name_output = '/Users/chuan/Research/Dataset/data/deepdream/mrf/';
format = '.png';
if(~exist(name_model, 'file'))
    warning('Pretrained model not found\n');
end
net = load(name_model);

name_img = ['/Users/chuan/Research/Dataset/data/deepdream/city.jpg'];


return;

% 
% % parameters
% opts.num_layers = 35;
% opts.octave_num = 3;
% opts.octave_scale = 2;
% opts.num_iter = 10;
% opts.lr = 2.5; % learning rate
% 
% opts.up_scale = 1;
% opts.average = 128; % image mean
% opts.bound = 255 - opts.average;
% opts.channel = 3;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Initialization
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% im_fullres = imread(name_img);
% im_fullres = imresize(im_fullres, [64 * 4, 64 * 3]);
% net.normalization.averageImage = single(ones(size(im_fullres)) * opts.average);
% net.normalization.imageSize = size(im_fullres);
% opts.normalize = get_cnn_normalize(net.normalization) ;
% opts.denormalize = get_cnn_denormalize(net.normalization) ;
% opts.imgSize = net.normalization.imageSize;
% net.layers = net.layers(1, 1:opts.num_layers);
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % dream
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x_ini = im_fullres;
% 
% % only keep the low frequency info
% x_ini = imresize(imresize(x_ini, 1/opts.up_scale, 'bicubic'), [size(x_ini, 1), size(x_ini, 2)], 'bicubic');
% x_ini = opts.normalize(x_ini);
% 
% % build pyramids
% octaves_img = cell(1, opts.octave_num);
% for i_oct = 1:opts.octave_num
%     octaves_img{1, i_oct} = imresize(x_ini, 1/opts.octave_scale^(i_oct - 1));
% end
% 
% res_ini = vl_simplenn(net, x_ini);
% 
% % render the feature patch and response
% [row, col] = ind2sub(size(res(end).x(:, :, 1)), sel_feature);
% opts.patch_mrf_size = 3;
% opts.stride_mrf_target = 1;
% scaler2fullres = 16;
% 
% coord_mrf_target = [];
% [coord_mrf_target(:, :, 1), coord_mrf_target(:, :, 2)] = meshgrid([(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_target:size(res(end).x, 2) - (opts.patch_mrf_size - 1)/2 - 1, size(res(end).x, 2) - (opts.patch_mrf_size - 1)/2], ...
%     [(opts.patch_mrf_size - 1)/2 + 1:opts.stride_mrf_target:size(res(end).x, 1) - (opts.patch_mrf_size - 1)/2 - 1, size(res(end).x, 1) - (opts.patch_mrf_size - 1)/2]);
% coord_mrf_target = reshape(coord_mrf_target, [], 2);
% coord_mrf_target_fullres =  coord_mrf_target * scaler2fullres - scaler2fullres/2 + 1;
%         
% 
% 
% 
% % % set a clear target here, so the top level derivatives can be correctly
% % % computed.
% % load ref.mat;
% % target_x = response_ref;
% % 
% % % res = vl_simplenn(net, x_ini);
% % % target_x = res(end).x;
% 
% % response_target = reshape(target_x, size(target_x, 3), []);
% % figure;
% % hold on;
% % for i = 1:size(response_target, 1)
% %     plot([1:size(response_target, 2)], response_target(i, :));
% % end
% % axis([0 size(response_target, 2) -2500 2500]);
% 
% for i_oct = 1:opts.octave_num
%     
%     if i_oct  == 1
%         x = octaves_img{1, opts.octave_num - i_oct + 1};
%     else
%         x = imresize(x, [size(octaves_img{1, opts.octave_num - i_oct + 1}, 1), size(octaves_img{1, opts.octave_num - i_oct + 1}, 2)]);
%     end
%     
%     for i_iter = 1:opts.num_iter
%         
%         res = vl_simplenn(net, x);
%         res(end).dzdx = res(end).x;
%         
% %         res(end).dzdx = target_x - res(end).x;  % L2 optimization
%         
%         res = vl_simplenn(net, x, res(end).dzdx);
%         mag_mean = mean(mean(mean(mean(abs(res(1).dzdx))))) + eps(1);     
%         x = x + (opts.lr * res(1).dzdx) / mag_mean;
%         
% %         x(x < -opts.bound) = -opts.bound;
% %         x(x > opts.bound) = opts.bound;
%         
%         im_cnn = x + opts.average;
%         
%         imwrite(im_cnn/255, [name_output 'output_oct_' num2str(i_oct) '_' num2str(i_iter) format]);
%         
%     end
%     
% end

sel_feature = 8;
sel_layer = 35;
[row, col] = ind2sub(size(res(sel_layer).x(:, :, 1)), sel_feature);

feature_start = reshape(res_ini(sel_layer).x(row, col, :), 1, []); 
% feature_start(feature_start < 0) = 0;
im_start = 0 * im_fullres;
im_start(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :) = ...
im_fullres(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :);                
      
feature_end = reshape(res(sel_layer).x(row, col, :), 1, []); 
% feature_end(feature_end < 0) = 0;
im_end = 0 * im_fullres;
im_end(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :) = ...
im_cnn(coord_mrf_target_fullres(sel_feature, 2) - 24:coord_mrf_target_fullres(sel_feature, 2) + 23, coord_mrf_target_fullres(sel_feature, 1) - 24:coord_mrf_target_fullres(sel_feature, 1) + 23, :);                

close all;
figure;
imshow([im_start im_end]);

figure;
hold on;
plot([1:size(feature_start, 2)], feature_start, 'r');
plot([1:size(feature_end, 2)], feature_end, 'b');
axis([0 size(feature_start, 2) -1500 1500]);

return;

idx = [1:size(feature_end, 2)];
idx = idx(feature_end > 0);
render_margin  = 5;
num_candidate = 10;
for i_layer = opts.num_layers:opts.num_layers
    if ~isempty(net.layers{1, i_layer}.features)
        num_feature = size(net.layers{1, i_layer}.features, 2);
        % make full size
        h = size(net.layers{1, i_layer}.features{1, 1}{1, 1}, 1);
        w = size(net.layers{1, i_layer}.features{1, 1}{1, 1}, 2);
        
        
        for i_feature_ = 1:size(idx, 2)
            im_out = ones(h, w * (num_candidate + 1) + render_margin * (num_candidate), 3);
            i_feature = idx(i_feature_);
            im_out(1:h, 1:w, :) = net.layers{1, i_layer}.feature_mean{1, i_feature};
            for i_patch = 1:size(net.layers{1, i_layer}.features{1, i_feature}, 2)
                im_out(1:h, (i_patch) * (w + render_margin) + 1:(i_patch) * (w + render_margin) + w, :) = net.layers{1, i_layer}.features{1, i_feature}{1, i_patch};
            end
            figure;
            imshow(im_out);
        end

%         imwrite(im_out, ['visual_layer_' num2str(i_layer) '_top_' num2str(num_candidate) '_img_' num2str(num_img) '.png']);
    else
    end
end
% return;

% response_dream = reshape(res(end).x, size(res(end).x, 3), []);
% figure;
% hold on;
% for i = 1:size(response_dream, 1)
%     plot([1:size(response_dream, 2)], response_dream(i, :));
% end
% axis([0 size(response_dream, 2) -2500 2500]);
% mean_dream = mean(response_dream(:));
% std_dream = std(response_dream(:));

return;

response_ref = res(end).x;
save('ref.mat', 'response_ref');
return;

response_ref = reshape(res(end).x, size(res(end).x, 3), []);
figure;
hold on;
for i = 1:size(response_ref, 1)
    plot([1:size(response_ref, 2)], response_ref(i, :));
end
axis([0 size(response_ref, 2) -500 500]);
mean_ref = mean(response_ref(:));
std_ref = std(response_ref(:));

for i_oct = 1:opts.octave_num
    
    if i_oct  == 1
        x = octaves_img{1, opts.octave_num - i_oct + 1};
    else
        x = imresize(x, [size(octaves_img{1, opts.octave_num - i_oct + 1}, 1), size(octaves_img{1, opts.octave_num - i_oct + 1}, 2)]);
    end
    
    for i_iter = 1:opts.num_iter
        
        res = vl_simplenn(net, x);
        res(end).dzdx = 0 * res(end).x;
        res(end).dzdx(:, :, opts.feature_range) = res(end).x(:, :, opts.feature_range);
        %         res(end).dzdx(res(end).dzdx < 0) = 0;
        res = vl_simplenn(net, x, res(end).dzdx);
        
        mag_mean = mean(mean(mean(mean(abs(res(1).dzdx))))) + eps(1);
        
        x = x + (opts.lr * res(1).dzdx) / mag_mean;
        
        x(x < -opts.bound) = -opts.bound;
        x(x > opts.bound) = opts.bound;
        
        im_cnn = x + opts.average;
        
        imwrite(im_cnn/255, [name_output 'output_oct_' num2str(i_oct) '_' num2str(i_iter) format]);
        
    end
    
end

response_dream = reshape(res(end).x, size(res(end).x, 3), []);
figure;
hold on;
for i = 1:size(response_dream, 1)
    plot([1:size(response_dream, 2)], response_dream(i, :));
end
axis([0 size(response_dream, 2) -500 500]);
mean_dream = mean(response_dream(:));
std_dream = std(response_dream(:));

return;

% target = reshape(response_dream, size(res(end).x));
% opts.num_iter_rec = 30;
% opts.lr_rec = 0.05;
% for i_iter = 1:opts.num_iter_rec
%     
% end

% normalize
response_norm = (response_dream + (mean_ref - mean_dream)) * (std_ref/ std_dream);
figure;
hold on;
for i = 1:size(response_norm, 1)
    plot([1:size(response_norm, 2)], response_norm(i, :));
end
axis([0 size(response_norm, 2) -500 500]);


% reconstruct image from response_norm
target = reshape(response_norm, size(res(end).x));
opts.num_iter_rec = 30;
opts.lr_rec = 0.5;
for i_iter = 1:opts.num_iter_rec
    res = vl_simplenn(net, x);
    res(end).dzdx = target - res(end).x;
    res = vl_simplenn(net, x, res(end).dzdx);
    mag_mean = mean(mean(mean(mean(abs(res(1).dzdx))))) + eps(1);
    x = x + (opts.lr_rec * res(1).dzdx) / mag_mean;
    x(x < -opts.bound) = -opts.bound;
    x(x > opts.bound) = opts.bound;
    im_cnn = x + opts.average;
    imwrite(im_cnn/255, [name_output 'rec_' num2str(i_iter) format]);
end

response_rec = reshape(res(end).x, size(res(end).x, 3), []);
figure;
hold on;
for i = 1:size(response_rec, 1)
    plot([1:size(response_rec, 2)], response_rec(i, :));
end
axis([0 size(response_rec, 2) -500 500]);
mean_dream = mean(response_rec(:));
std_dream = std(response_rec(:));



