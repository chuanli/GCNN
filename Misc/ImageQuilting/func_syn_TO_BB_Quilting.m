function result = func_syn_TO_BB_Quilting(data, P_TO_BB)

% reconstruct the full resolution
% find the patches in the original image
%         im_input_fullres
num_patch = data.num_col * data.num_row;
num_col = data.num_col;
num_row = data.num_row;
num_col_syn = data.num_col_syn;
num_row_syn = data.num_row_syn;
patch2patch = data.patch2patch;

w = data.w/data.scaler;
dist_center_input = data.dist_center_input/data.scaler;
dist_center_syn = data.dist_center_syn/data.scaler;

patch_input_gray = zeros(num_patch, w * w);
patch_input_red = zeros(num_patch, w * w);
patch_input_green = zeros(num_patch, w * w);
patch_input_blue = zeros(num_patch, w * w);

im_input_gray_fullres = data.im_input_gray_fullres;

max_col = (num_col - 1) * dist_center_input + w;
max_row = (num_row - 1) * dist_center_input + w;
pad_col = max_col - size(im_input_gray_fullres, 2);
pad_row = max_row - size(im_input_gray_fullres, 1);

im_input_gray_fullres = padarray(P_TO_BB.im_input_gray_fullres, [max(0, pad_row) max(0, pad_col)], 'replicate', 'post');
im_input_fullres = padarray(P_TO_BB.im_input_fullres,  [max(0, pad_row) max(0, pad_col)], 'replicate', 'post');


for i = 1:num_col
    start_x = (i - 1) * dist_center_input + 1;
    end_x = (i - 1) * dist_center_input + w;
    for j = 1:num_row
        start_y = (j - 1) * dist_center_input + 1;
        end_y = (j - 1) * dist_center_input + w;
        idx = (i - 1) * num_row + j;
        patch_input_gray(idx, :) = reshape(im_input_gray_fullres(start_y:end_y, start_x:end_x, 1), 1, []);
        patch_input_red(idx, :) = reshape(im_input_fullres(start_y:end_y, start_x:end_x, 1), 1, []);
        patch_input_green(idx, :) = reshape(im_input_fullres(start_y:end_y, start_x:end_x, 2), 1, []);
        patch_input_blue(idx, :) = reshape(im_input_fullres(start_y:end_y, start_x:end_x, 3), 1, []);
    end
end

im_syn_gray_fullres = zeros(dist_center_syn * (num_row_syn - 1) + w, dist_center_syn * (num_col_syn - 1) + w);
im_syn_color_fullres = zeros(dist_center_syn * (num_row_syn - 1) + w, dist_center_syn * (num_col_syn - 1) + w, 3);
im_syn_color_fullres2 = zeros(dist_center_syn * (num_row_syn - 1) + w, dist_center_syn * (num_col_syn - 1) + w, 3);

im_syn_count = 0 * im_syn_gray_fullres;
for i = 1:num_col_syn
    for j = 1:num_row_syn
        idx = (i - 1) * num_row_syn + j;
        [row, col] = ind2sub([num_row, num_col], patch2patch(1, idx));
        K_gray = reshape(patch_input_gray(patch2patch(1, idx), :), w, w);
        K_red = reshape(patch_input_red(patch2patch(1, idx), :), w, w);
        K_green = reshape(patch_input_green(patch2patch(1, idx), :), w, w);
        K_blue = reshape(patch_input_blue(patch2patch(1, idx), :), w, w);
        
        im_syn_gray_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 1) = im_syn_gray_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 1) + K_gray;
        im_syn_color_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 1) = im_syn_color_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 1) + K_red;
        im_syn_color_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 2) = im_syn_color_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 2) + K_green;
        im_syn_color_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 3) = im_syn_color_fullres((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 3) + K_blue;
        im_syn_color_fullres2((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 1) = K_red;
        im_syn_color_fullres2((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 2) = K_green;
        im_syn_color_fullres2((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w, 3) = K_blue;
        
        im_syn_count((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w) = im_syn_count((j - 1) * dist_center_syn + 1:(j - 1) * dist_center_syn + w, (i - 1) * dist_center_syn + 1:(i - 1) * dist_center_syn + w) + 1;
    end
end

im_syn_color_fullres(:, :, 1) = im_syn_color_fullres(:, :, 1)./im_syn_count;
im_syn_color_fullres(:, :, 2) = im_syn_color_fullres(:, :, 2)./im_syn_count;
im_syn_color_fullres(:, :, 3) = im_syn_color_fullres(:, :, 3)./im_syn_count;
im_syn_gray_fullres = im_syn_gray_fullres./im_syn_count;


result.average = im_syn_color_fullres;
result.padding = im_syn_color_fullres2;
result.average_gray = im_syn_gray_fullres;


im_syn_gray_fullres_reference = im_syn_gray_fullres;

filt_w = 4;
smooth_filt = binomialFilter(filt_w)*binomialFilter(filt_w)';
dest_mask = logical(zeros(size(im_syn_gray_fullres)));
ww = dist_center_syn;
border = w - dist_center_syn;

im_syn_gray_fullres = NaN * ones(size(im_syn_gray_fullres));
im_syn_red_fullres = NaN * ones(size(im_syn_gray_fullres));
im_syn_green_fullres = NaN * ones(size(im_syn_gray_fullres));
im_syn_blue_fullres = NaN * ones(size(im_syn_gray_fullres));

work_im_rgb = ones(size(im_syn_color_fullres));
work_im_rgb2 = work_im_rgb;

if P_TO_BB.flag_render == 1
    figure(1);
    colormap(gray);
    subplot(1,2,1);
    imshow(im_syn_gray_fullres);
    title('Current dest');
    subplot(1,2,2);
    imshow(work_im_rgb);
    title('Result');
    truesize;
    drawnow;
end


for i = 1:P_TO_BB.IQ_step:num_col_syn
    for j = 1:P_TO_BB.IQ_step:num_row_syn
% for i = 1:1
%     for j = 1:2
        ii = (j - 1) * ww + 1;
        jj = (i - 1) * ww + 1;
        if (~all(all(dest_mask(ii:ii+ww+border-1, jj:jj+ww+border-1))))
            template = im_syn_gray_fullres(ii:ii+ww+border-1, jj:jj+ww+border-1);
            if P_TO_BB.IQ_color_mode == 1
                template_red = im_syn_red_fullres(ii:ii+ww+border-1, jj:jj+ww+border-1);
                template_green = im_syn_green_fullres(ii:ii+ww+border-1, jj:jj+ww+border-1);
                template_blue = im_syn_blue_fullres(ii:ii+ww+border-1, jj:jj+ww+border-1);
            end
            
            idx = (i - 1) * num_row_syn + j;
            K_gray = reshape(patch_input_gray(patch2patch(1, idx), :), w, w);
            K_rgb = cat(3, reshape(patch_input_red(patch2patch(1, idx), :), w, w), reshape(patch_input_green(patch2patch(1, idx), :), w, w), reshape(patch_input_blue(patch2patch(1, idx), :), w, w));
            
            [row, col] = ind2sub([num_row, num_col], patch2patch(1, idx));
            blend_mask = logical(zeros(size(template)));
            
            
            %
            if P_TO_BB.IQ_color_mode == 1
                err_sq = ((template_red - K_rgb(:, :, 1)).^2 + (template_green - K_rgb(:, :, 2)).^2 + (template_blue - K_rgb(:, :, 3)).^2)/3;
            else
                err_sq = (template - K_gray).^2;
            end
            %                         disp(['Error: ' num2str(sum(sum(err_sq))) ', i: ' num2str(i) ', j: ' num2str(j)]);
            if (ii > 1 & jj > 1)
                blend_mask = dpmain(err_sq, border);
            elseif (ii == 1 & jj == 1)
                ;
            elseif (ii == 1)
                blend_mask(:,1:border) = dp(err_sq(:,1:border));
            else
                blend_mask(1:border,:) = blend_mask(1:border,:) | dp(err_sq(1:border,:)')';
            end
            
            blend_mask = rconv2(double(blend_mask),smooth_filt); % Do blending
            blend_mask_rgb = repmat(blend_mask,[1 1 3]);
            template(isnan(template)) = 0;
            if P_TO_BB.IQ_color_mode == 1
                template_red(isnan(template_red)) = 0;
                template_green(isnan(template_green)) = 0;
                template_blue(isnan(template_blue)) = 0;
            end
            
            work_im_rgb(ii:ii+ww+border-1,jj:jj+ww+border-1,:) ...
                = work_im_rgb(ii:ii+ww+border-1,jj:jj+ww+border-1,:).*blend_mask_rgb...
                + K_rgb.*(1-blend_mask_rgb);
            
            work_im_rgb2(ii:ii+ww+border-1,jj:jj+ww+border-1,:) ...
                = work_im_rgb2(ii:ii+ww+border-1,jj:jj+ww+border-1,:)...
                .*(blend_mask_rgb>=0.95)...
                + K_rgb.*((1-blend_mask_rgb)>=0.95);
            
            im_syn_gray_fullres(ii:ii+ww+border-1,jj:jj+ww+border-1) ...
                = template.*blend_mask + K_gray.*(1-blend_mask);
            
            if P_TO_BB.IQ_color_mode == 1
                im_syn_red_fullres(ii:ii+ww+border-1,jj:jj+ww+border-1) ...
                    = template_red.*blend_mask + K_red.*(1-blend_mask);
                
                im_syn_green_fullres(ii:ii+ww+border-1,jj:jj+ww+border-1) ...
                    = template_green.*blend_mask + K_green.*(1-blend_mask);
                
                im_syn_blue_fullres(ii:ii+ww+border-1,jj:jj+ww+border-1) ...
                    = template_blue.*blend_mask + K_blue.*(1-blend_mask);
            end
        end
        
    end
    if P_TO_BB.flag_render == 1
        figure(1);
        subplot(1,2,1);
        imshow(im_syn_gray_fullres);
        subplot(1,2,2);
        imshow(work_im_rgb);
        drawnow;
    end
end

result.work_im_rgb = work_im_rgb;
result.work_im_rgb2 = work_im_rgb2;
result.im_syn_gray_fullres = im_syn_gray_fullres;
