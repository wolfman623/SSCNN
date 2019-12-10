% demo:
% color boundary detection with oriented Double-opponent cells.
% paper in CVPR 2013:
% Kaifu Yang, Shaobing Gao, Chaoyi Li, and Yongjie Li*.
% Efficient Color Boundary Detection with Color-opponent Mechanisms. CVPR, 2013.
%
% Contact:
% Visual Cognition and Computation Laboratory(VCCL),
% Key Laboratory for Neuroinformation of Ministry of Education,
% School of Life Science and Technology,
% University of Electronic Science and Technology of China, Chengdu, 610054, China
% Website: http://www.neuro.uestc.edu.cn/vccl/computation_projects.html
%
% Kaifu Yang <yang_kf@163.com>
% March 2013
%=========================================================================%

% clc;  clear;

% parameters setting
angles = 16;
sigma = 0.5;
weights = 0;

% read original image
name = 'map2';
path = ['.\..\dataset\' name '\'];
map = double(imread([path name '.bmp']))./255;
% figure;imshow(map,[]);

% tic
CO = COBoundary(map,sigma,angles,weights);
% toc

save([path 'CO.mat'], 'CO');
% figure;imshow(CO,[]);
% imwrite(CO,[path name '_boundary.bmp']);

fprintf(2,'======== THE END ========\n');
%=========================================================================%
