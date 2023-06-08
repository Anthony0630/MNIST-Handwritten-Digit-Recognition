function [features] = GetFeatures(image)
%对图像进行特征提取，features为提取出的特征矩阵
%采用网格特征提取法即将图像分割成若干个10*10的子矩阵，
%统计每个子矩阵中白色像素的个数，返回特征矩阵
[a, b] = size(image);
m = a/10;
n = b/10; %获取分割网格的维数m*n
features = zeros(m,n); %初始化特征矩阵
for i = 1 : m
    for j = 1 : n
        mat = image(10*(i-1)+1:10*i,10*(j-1)+1:10*j);
        features(i,j) = length(find(mat == 1));  %统计当前分割块中白色像素的个数
    end
end
features = features(:)'; %将features转换为行向量