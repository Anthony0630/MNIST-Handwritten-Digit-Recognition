clc
images = loadMNISTImages('/Users/chenghaidong/Desktop/Machine Learning/ML课程大作业/minist/train-images.idx3-ubyte');
labels = loadMNISTLabels('/Users/chenghaidong/Desktop/Machine Learning/ML课程大作业/minist/train-labels.idx1-ubyte');
images = images';  %读入images和labels矩阵
[m, n] = size(labels);
for i = 1 : m
    if labels(i,1) == 0
        labels(i,1) = 10;
    end
end      %将所有为0的label改为10，代表共有10个label
images = images(1:5000,:); %取前5000个为训练集
labels = labels(1:5000,:);
[a, b] = size(images);  %获取训练集的维数
%对所有图像进行二值化处理 
for i = 1 : a
    for j = 1 : b
        if images(i,j) > 0
            images(i,j) = 1;
        end
    end
end
train_features = zeros(a,35); %初始化特征向量
%对所有图像进行遍历切割
for i = 1 : a
    sample = images(i,:);
    sample = reshape(sample,28,28); %将行向量恢复成图片
    Cutted_sample = Cut(sample,50,70); %将图片进行切割，统一大小
    train_features(i,:) = GetFeatures(Cutted_sample); %获取当前图像的特征行向量
end
data = [train_features,labels];
train_data = data_standardlize(data);
NeuronNum = 50;
OutputNum = 10;
train_number = 1000;
[theta, b1, b2, w] = BP_Train(train_data, NeuronNum, OutputNum, train_number)