function result = boundary_cover_img(img,boundary)
imgr = img(:,:,1);
imgg = img(:,:,2);
imgb = img(:,:,3);
imgr(~boundary) = 255;
imgg(~boundary) = 0;
imgb(~boundary) = 0;
result = cat(3,imgr,imgg,imgb);
end