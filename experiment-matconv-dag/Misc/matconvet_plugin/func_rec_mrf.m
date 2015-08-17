function rec = func_rec_mrf(rec_sz, list_match, synthesis_bottom_loc, example_bottom_loc, example_img_id, var_id, value_example)
rec = zeros(rec_sz);
count = 0 * rec(:, :, 1);

for i_patch = 1:size(list_match, 2)
    rec(synthesis_bottom_loc(i_patch, 2):synthesis_bottom_loc(i_patch, 4), synthesis_bottom_loc(i_patch, 1):synthesis_bottom_loc(i_patch, 3), :) = rec(synthesis_bottom_loc(i_patch, 2):synthesis_bottom_loc(i_patch, 4), synthesis_bottom_loc(i_patch, 1):synthesis_bottom_loc(i_patch, 3), :) ...
        + value_example{1, example_img_id(list_match(1, i_patch))}(var_id).value(example_bottom_loc(list_match(1, i_patch), 2):example_bottom_loc(list_match(1, i_patch), 4), example_bottom_loc(list_match(1, i_patch), 1):example_bottom_loc(list_match(1, i_patch), 3), :);
    count(synthesis_bottom_loc(i_patch, 2):synthesis_bottom_loc(i_patch, 4), synthesis_bottom_loc(i_patch, 1):synthesis_bottom_loc(i_patch, 3), :) = count(synthesis_bottom_loc(i_patch, 2):synthesis_bottom_loc(i_patch, 4), synthesis_bottom_loc(i_patch, 1):synthesis_bottom_loc(i_patch, 3), :) + 1;
end
for i_depth = 1:size(rec, 3)
    rec(:, :, i_depth) = rec(:, :, i_depth) ./ count;
end

