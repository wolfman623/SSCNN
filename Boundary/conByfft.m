function convRlt = conByfft(img,tpl)
% conv2 throuht fft2...
% same as "imfiter" function with the border parameter: 'symmetric'...
% the output "convRlt" has the same size as input image(img)...
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

img = double(img);
[Nr,Nc,Ns] = size(img);
[nr,nc]    = size(tpl);
nrE = floor(double(nr)/2);
ncE = floor(double(nc)/2);
imgExpan2 = zeros(Nr+4*nrE,Nc+4*ncE,Ns);
tplExpan  = zeros(Nr+4*nrE,Nc+4*ncE);
tplExpan(1:nr,1:nc)  = tpl;

for i = 1:Ns
    imgExpan1(:,:,i) = padarray(img(:,:,i),[nrE ncE],'symmetric','both');
    imgExpan2(1:Nr+2*nrE,1:Nc+2*ncE,i) = imgExpan1(:,:,i);
    convRlt(:,:,i) = real(ifft2(fft2(imgExpan2(:,:,i)).*fft2(tplExpan)));
end

convRlt = convRlt(2*nrE+1:2*nrE+Nr,2*ncE+1:2*ncE+Nc);
%=========================================================================%
