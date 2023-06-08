clc
images = loadMNISTImages('/Users/chenghaidong/Desktop/Machine Learning/ML�γ̴���ҵ/minist/train-images.idx3-ubyte');
labels = loadMNISTLabels('/Users/chenghaidong/Desktop/Machine Learning/ML�γ̴���ҵ/minist/train-labels.idx1-ubyte');
images = images';  %����images��labels����
[m, n] = size(labels);
for i = 1 : m
    if labels(i,1) == 0
        labels(i,1) = 10;
    end
end      %������Ϊ0��label��Ϊ10��������10��label
images = images(1:5000,:); %ȡǰ5000��Ϊѵ����
labels = labels(1:5000,:);
[a, b] = size(images);  %��ȡѵ������ά��
%������ͼ����ж�ֵ������ 
for i = 1 : a
    for j = 1 : b
        if images(i,j) > 0
            images(i,j) = 1;
        end
    end
end
train_features = zeros(a,35); %��ʼ����������
%������ͼ����б����и�
for i = 1 : a
    sample = images(i,:);
    sample = reshape(sample,28,28); %���������ָ���ͼƬ
    Cutted_sample = Cut(sample,50,70); %��ͼƬ�����иͳһ��С
    train_features(i,:) = GetFeatures(Cutted_sample); %��ȡ��ǰͼ�������������
end
data = [train_features,labels];
train_data = data_standardlize(data);
NeuronNum = 50;
OutputNum = 10;
train_number = 1000;
[theta, b1, b2, w] = BP_Train(train_data, NeuronNum, OutputNum, train_number)