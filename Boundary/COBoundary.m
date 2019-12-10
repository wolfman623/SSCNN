function fb = COBoundary(map,sigma,angles,weights)
% function fb = COBoundary(map,sigma,angles,weights)
% inputs:
%         map ------ RGB color map.
%         sigma ---- Local scale (the size of cones' RF).
%         angles --- the number of orientation.
%         weights -- one of the cone weights(the other one is 1).
% outputs:
%         fb  ------ final soft boundary
%
% Main function for performing boundary detection system based 
% on Color-Opponent(CO) in paper:
% Kaifu Yang, Shaobing Gao, Chaoyi Li, and Yongjie Li*.
% Efficient Color Boundary Detection with Color-opponent Mechanisms. CVPR, 2013.
% 
% Contact:
% Visual Cognition and Computation Laboratory(VCCL),
% Key Laboratory for NeuroInformation of Ministry of Education,
% School of Life Science and Technology(SLST), 
% University of Electrical Science and Technology of China(UESTC).
% Address: No.4, Section 2, North Jianshe Road,Chengdu,Sichuan,P.R.China, 610054
% Website: http://www.neuro.uestc.edu.cn/vccl/computation_projects.html
% ���ӿƼ���ѧ��������ѧ�뼼��ѧԺ��
% ����Ϣ�������ص�ʵ���ң��Ӿ���֪�������
% �й����Ĵ����ɶ������豱·����4�ţ�610054

% ���/Kaifu Yang <yang_kf@163.com>;
% ������/Yongjie Li <liyj@uestc.edu.cn>;
% March 2013
%
%========================================================================%

if nargin < 4, weights= -0.6; end
if nargin < 3,  angles = 8;  end

% obtain the final response
[Res theta] = resDO(map,sigma,angles,weights);
Re = Res./max(Res(:));

% non-max suppression...
theta = (theta-1)*pi/angles;
theta = mod(theta+pi/2,pi);
fb = nonmax(Re,theta);
fb = max(0,min(1,Re));

% mask out 1-pixel border where nonmax suppression fails
fb(1,:) = 0;
fb(end,:) = 0;
fb(:,1) = 0;
fb(:,end) = 0;
%========================================================================%
