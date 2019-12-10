% This script is for linear element extraction, based on COGF
name = 'map2';
path = ['.\..\dataset\' name '\'];
img = imread([path name '.bmp']);
imgr = img(:,:,1);
imgg = img(:,:,2);
imgb = img(:,:,3);
% figure,imshow(img);
img = double(img);
ndir = 3;
tic;
for k = -2^(ndir):2^(ndir)
   dd(:,:,1,k+2^(ndir)+1)= dshear(img(:,:,1),k,0,ndir,1);
   
   dd(:,:,2,k+2^(ndir)+1)= dshear(img(:,:,2),k,0,ndir,1);
   
   dd(:,:,3,k+2^(ndir)+1)= dshear(img(:,:,3),k,0,ndir,1);
end
result = zeros(size(img(:,:,1)));
for k = -2^(ndir):2^(ndir)

img = cat(3,dd(:,:,1,k+2^(ndir)+1),dd(:,:,2,k+2^(ndir)+1),dd(:,:,3,k+2^(ndir)+1));
img = double(img);
wtemp1 = -fspecial('gaussian',[1 4],3);    %Theta2
wtemp2 = fspecial('gaussian',[1,3],1);    %Theta1
wtemp2(find(wtemp2<0.001)) = [];
temp = length(wtemp1) + length(wtemp2);
w = zeros(temp,temp);
templ = ceil(temp/2);
w(templ,:) = [wtemp1(1:length(wtemp1)/2) wtemp2 wtemp1(length(wtemp1)/2+1:end)];
w(:,templ) = [wtemp1(1:length(wtemp1)/2) wtemp2 wtemp1(length(wtemp1)/2+1:end)];
w(templ,templ) = w(templ,templ)*2;

for i = 1:3
    layer = img(:,:,i);
    layer = imfilter(layer,w');
    img(:,:,i) = layer;
end
[heigh,width] = size(img(:,:,1));
layer = zeros(heigh,width);
for i = 1:heigh
    for j = 1:width
        layer(i,j) = sum(img(i,j,:));
    end
end
layer = dshear(layer,k,0,ndir,0);
result = result + layer;
% result = max(result,layer);
% figure,imshow(layer,[]);
% figure,imshow(slayer)
end
toc;
sresult = result<-400;
% figure,imshow(sresult);
sresult = bwareaopen(sresult,8);
% figure,imshow(sresult);
imgr(find(~sresult)) = 255;
imgg(find(~sresult)) = 255;
imgb(find(~sresult)) = 255;
img = cat(3,imgr,imgg,imgb);
% figure,imshow(img);
imwrite(img,[path name 'b.bmp']);
% imwrite(sresult,[path name 'b_bw.bmp']);
