function g = gaus(sgm)
% function g = gaus(sgm)
% 2-D gauss function
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

nr = 15*sgm;
nc = 15*sgm;
[x,y] = meshgrid(linspace(-nc/2, nc/2, nc), linspace(nr/2, -nr/2, nr));
g = exp(-(x.^2+y.^2)/(2*(sgm)^2));
g = g/sum(g(:));

%=========================================================================%
