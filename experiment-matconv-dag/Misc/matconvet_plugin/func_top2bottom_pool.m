function bottom_loc = func_top2bottom_pool(top, top_loc, block)
bottom_loc = [];
if block.stride(1) == 0
    bottom_loc = top_loc;
else
    bottom_loc = (top_loc - 1) * block.stride(1) + 1;
end