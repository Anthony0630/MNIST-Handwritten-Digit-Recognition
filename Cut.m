function [imageout] = Cut(image,mout,nout) 
%��ͼ���и������ʻ����У��ٶ�ͼ�����һ�ζ�ֵ������,�ٽ���һ�αղ���
%�����mout,noutΪ��10��
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
    [a, b] = size(imageout);  %��ȡѵ������ά��
    %��ͼ����ж�ֵ������
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
    %��ͼ����бղ���
    H = strel('disk',2);
    imageout = imclose(imageout,H);
end
