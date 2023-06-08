images = loadMNISTImages('C:\Users\Anthony\Desktop\201883250055-程海东-机器学习课程报告\minist\train-images.idx3-ubyte');
labels = loadMNISTLabels('C:\Users\Anthony\Desktop\201883250055-程海东-机器学习课程报告\minist\train-labels.idx1-ubyte');
images = images';  %读入images和labels矩阵
[m, n] = size(labels);
b1 = load('b1_final.mat');
b1_final = b1.b1;
b2 = load('b2_final.mat');
b2_final = b2.b2;
theta = load('theta_final.mat');
theta_final = theta.theta;
w = load('w_final.mat');
w_final = w.w;   %读入训练好的网络权重
for i = 1 : m
    if labels(i,1) == 0
        labels(i,1) = 10;
    end
end      %将所有为0的label改为10，代表共有10个label
images = images(1001:1200,:); %取200个为测试集
labels = labels(1001:1200,:);
label = labels';
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
test_Y = zeros(200,10);
for num = 1 : 200
    current_label = zeros(1,10);
    current_label(1,label(1,num)) = 1;
    X = train_data(num,1:end-1);
    T = sigmoid(X * theta_final + b1_final);  %计算得出隐层T的神经元节点值
    Y(num,:) = sigmoid(T * w_final + b2_final);    %计算得出输出层Y的神经元节点值
%     for i = 1 : 10
%         if Y(num,i) >= 0.5
%             Y(num,i) = 1;
%         end
%         if Y(num,i) < 0.5
%             Y(num,i) = 0;
%         end
%     end
end
test_label = zeros(200,1);
for i = 1 : 200
    [max_val, index] = max(Y(i,:));
    test_label(i,1) = index;
    if test_label(i,1) == 10
        test_label(i,1) = 0;
    end
end
for i = 1 : 200
    if labels(i,1) == 10
        labels(i,1) = 0;
    end
end
result = [test_label,labels]
right = 0;
for i = 1 : 200
    if result(i,1) == result(i,2)
        right = right + 1;
    end
end
right
percentage = right/200;
percentage = [num2str(100*percentage),'%']
