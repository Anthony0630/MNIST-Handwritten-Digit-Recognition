function [features] = GetFeatures(image)
%��ͼ�����������ȡ��featuresΪ��ȡ������������
%��������������ȡ������ͼ��ָ�����ɸ�10*10���Ӿ���
%ͳ��ÿ���Ӿ����а�ɫ���صĸ�����������������
[a, b] = size(image);
m = a/10;
n = b/10; %��ȡ�ָ������ά��m*n
features = zeros(m,n); %��ʼ����������
for i = 1 : m
    for j = 1 : n
        mat = image(10*(i-1)+1:10*i,10*(j-1)+1:10*j);
        features(i,j) = length(find(mat == 1));  %ͳ�Ƶ�ǰ�ָ���а�ɫ���صĸ���
    end
end
features = features(:)'; %��featuresת��Ϊ������