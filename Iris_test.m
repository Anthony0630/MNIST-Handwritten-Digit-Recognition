clc
data = load('/Users/chenghaidong/Desktop/Machine Learning/kNN/irisdataset.mat');
train_data = [data.irisdata,data.kind];
label = [data.kind]';
[s t] = size(train_data); %����s��������t-1������ά�ȣ����һ��Ϊlabel
%���й�һ������
Means = zeros(1,t-1); %��ʼ����ֵ����
Stds = zeros(1,t-1); %��ʼ����׼�����
for i = 1 : t-1
    Means(1,i) = mean(train_data(:,i));
    Stds(1,i) = std(train_data(:,i));
end
for i = 1 : s
    for j = 1 : t-1
        train_data(i,j) = (train_data(i,j) - Means(1,j))/Stds(1,j);
    end
end
NeuronNum = 100;
OutputNum = 3;
train_number = 1000;
[theta, b1, b2, w] = BP_Train(train_data, NeuronNum, OutputNum, train_number);
test_label = zeros(s,1);
for num = 1 : s
    current_label = zeros(1,OutputNum);
    current_label(1,label(1,num)) = 1;
    X = train_data(num,1:end-1);
    T = sigmoid(X * theta + b1);  %����ó�����T����Ԫ�ڵ�ֵ
    Y = sigmoid(T * w + b2);    %����ó������Y����Ԫ�ڵ�ֵ
    for i = 1 : OutputNum
        if Y(1,i) >= 0.5
            Y(1,i) = 1;
        end
        if Y(1,i) < 0.5
            Y(1,i) = 0;
        end
    end
    test_label(num,1) = find(Y,1);
end
test_label
    