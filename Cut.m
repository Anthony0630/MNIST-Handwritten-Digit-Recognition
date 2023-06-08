function [imageout] = Cut(image,mout,nout) 
%将图像切割到各边与笔画相切，再对图像进行一次二值化处理,再进行一次闭操作
%输出的mout,nout为整10数
    [m,n] = size(image);
    Left = n;
    Right = 1;
    Top = m;
    Bottom = 0;
    for i = 1 : n
        for j = 1 : m
            if image(j,i) > 0
                if i < Left
                    Left = i;
                end
                if i > Right
                    Right = i;
                end
                if j > Bottom
                    Bottom = j;
                end
                if j < Top
                    Top = j; 
                end
            end
        end 
    end
    imagecrop = imcrop(image,[Left,Top,Right-Left,Bottom-Top]);
    imageout = imresize(imagecrop,[mout nout]);
    [a, b] = size(imageout);  %获取训练集的维数
    %对图像进行二值化处理
    for i = 1 : a
        for j = 1 : b
            if imageout(i,j) <= 0
                imageout(i,j) = 0;
            end
            if imageout(i,j) > 0
                imageout(i,j) = 1;
            end
        end
    end
    %对图像进行闭操作
    H = strel('disk',2);
    imageout = imclose(imageout,H);
end
