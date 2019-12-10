function [Ch12 Ch21] = SingleOpponent(map,opponent,sigma,weights)
% function [Ch12 Ch21] = SingleOpponent(map,opponent,sigma,weights)
% Compute the responses of single-opponent cells
% inputs:
%         map ------ RGB color map.
%         opponent - opponent channel one of {'RG','BY'}.
%         sigma ---- Local scale (the size of Gaussion).
%         weights -- one of the cone weights(the other one is 1).
% outputs:
%         Ch12 --- the response in "channel1 + w * channel2 (e.g., R + w*G)"
%         Ch21 --- the response in "channel2 + w * channel1 (e.g., w*R + G)"
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

if nargin < 4, weights= -0.6; end

[rr cc d] = size(map);
if d~=3
    error('The input image must be a color image(3-D matrix)!\n');
end

% obtain each channel
R = map(:,:,1);   % Cone-L
G = map(:,:,2);   % Cone-M
B = map(:,:,3);   % Cone-S
Y = (R+G)/2;

% select the opponent channels
if strcmp(opponent,'RG')
    channel1 = R;  channel2 = G;
elseif strcmp(opponent,'BY')
    channel1 = B;  channel2 = Y;
else
    error('the opponent channel parameter must be one of {RG,BY}\n');
end

% compute the response of center-only single opponent cells
w1 = 1.0 * ones(rr,cc);
w2 = weights .* ones(rr,cc);

gau2D = gaus(sigma);
Ch1 = imfilter(channel1,gau2D,'conv','replicate');
Ch2 = imfilter(channel2,gau2D,'conv','replicate');
% Ch1 = channel1;
% Ch2 = channel2;

Ch12 = w1.*Ch1 + w2.*Ch2;   % Ch1+ Ch2-
Ch21 = w2.*Ch1 + w1.*Ch2;   % Ch1- Ch2+
%=========================================================================%
