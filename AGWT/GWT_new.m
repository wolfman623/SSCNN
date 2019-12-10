% tic;
name = 'map2';
path = ['.\..\dataset\' name '\'];
img = imread([path name '.bmp']);
img_line = imread([path name 'b.bmp']);
load([path 'CO.mat']);
boundary = CO;

%% load the linear elements
line_bw = ~im2bw(img_line,0.99);
area_bw = ~line_bw;
pic_thin = bwmorph(line_bw,'thin',inf);   

w = ones(3,3);
w(2,2) = 0;
JD = imfilter(double(pic_thin),w);
JD = (JD>2)&(pic_thin==1);

w = ones(3,3);
for i = 1:3
    JD = imdilate(JD,w);
end


se = ones(5,5);
AR = imerode(area_bw,se);
boundary_tmp = boundary(AR);
remove_boundary_intensity = graythresh(boundary_tmp);
% remove_boundary_intensity = mean(boundary_tmp);

pic_thin(JD) = 0;
% [label,label_num] = bwlabel(pic_thin,8);
boundary(pic_thin) = 0;
% boundary(AR) = 0;
remove_boundary_tmp = AR*remove_boundary_intensity;
boundary = boundary - remove_boundary_tmp;
boundary = max(boundary,0);

boundary(:,1) = 1;
boundary(:,end) = 1;
boundary(1,:) = 1;
boundary(end,:) = 1;
patch = watershed(boundary);

figure,imshow(img);
% figure,imshow(patch)
imwrite(double(patch),[path name '_superpix.bmp']);
boundary = logical(patch);
res = boundary_cover_img(img,boundary);
imwrite(res,[path name '_advanceGWT.bmp']);
figure, imshow(res);
% toc;