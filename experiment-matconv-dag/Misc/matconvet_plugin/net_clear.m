function net = net_clear(net)

% clear data
for i_vars = 1:size(net.vars, 2)
    net.vars(i_vars).value = [];
    net.vars(i_vars).der = [];
end
