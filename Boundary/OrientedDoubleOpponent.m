function [DO12 DO21] = OrientedDoubleOpponent(map,opponent,sigma,angles,weights)
% function [DO12 DO21] = OrientedDoubleOpponent(map,opponent,sigma,angles,weights)
% Compute the responses of oriented double-opponent cells on R-G or B-Y channel
% inputs:
%         map ------ RGB color map.
%         opponent - opponent channel one of {'RG','BY'}.
%         sigma ---- Local scale (the size of cones' RF).
%         angles --- the number of orientation.
%         weights -- one of the cone weights(the other one is 1).
% outputs:
%        DO12 --- responses of oriented double-opponent cells
%                 in "channel1 + w * channel2 (e.g., R + w*G)"
%        DO21 --- responses of oriented double-opponent cells
%                 in "channel2 + w * channel1 (e.g., w*R + G)"
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
%==========================================================================%

if nargin < 5, weights= -0.6; end
if nargin < 4,  angles = 8;  end

% compute responses of single-opponent cells
[Ch12 Ch21] = SingleOpponent(map,opponent,sigma,weights);

% construct RF of the oriented double-opponent
sig = 2*sigma;
DO12 = zeros(size(map,1),size(map,2),angles);
DO21 = zeros(size(map,1),size(map,2),angles);

fprintf(2,'[');

% Obtain the response with the filters in degree of [0 pi],
% and then taking the abslute value, which is same as rotating the  filters
% in [0 2*pi]
for i = 1:angles
    dgau2D = DivGauss2D(sig,(i-1)*pi/angles);
    S = sum(abs(dgau2D(:)));
    t1 = conByfft(Ch12,dgau2D/S); % Ch12
    DO12(:,:,i) = abs(t1);
    t2 = conByfft(Ch21,dgau2D/S); % Ch21
    DO21(:,:,i) = abs(t2);

    fprintf(2,'.');
end

fprintf(2,']\n');
%=========================================================================%
