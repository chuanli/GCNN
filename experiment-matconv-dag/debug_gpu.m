close all; clear all; clc;
load test;
x = imresize(x, 1);

path_matconvnet = '../third-party/matconvnet-dag/';
path_flann = '../third-party/flann/';
path_net = '../data/nets/';
path_input =  '../data/input/';
path_input_img_ini =  [path_input 'img_ini/'];
path_input_example_mrf =  [path_input 'example_mrf/'];
path_input_example_gd =  [path_input 'example_gd/'];
path_output = '../data/output/synthesis_matconvnet_dag_mrfb_gdt_flann_nocontrol/';
mkdir(path_output);
format_output = '.png';
run ([path_matconvnet 'matlab/vl_setupnn']) ;
addpath('./Misc/deep-goggle');
addpath('./Misc/matconvet_plugin');
addpath(genpath(path_flann));


name_input_net = 'imagenet-vgg-verydeep-19-dag.mat';
myset.net.end_var = 'pool1';
% myset.net.end_var = 'relu5_4';

netDAG = load([path_net name_input_net]) ;
net = dagnn.DagNN.loadobj(netDAG) ;
myset.net.num_vars = net.getVarIndex(myset.net.end_var);
for i = size(net.vars, 2):-1:myset.net.num_vars + 1
    net.removeLayer(net.layers(end).name);
end

net.conserveMemory = 0;

net.move('gpu');
net.eval({'data', x});
gpuDevice(1)

% vars = net.vars;
