images = loadMNISTImages('C:\Users\Anthony\Desktop\201883250055-�̺���-����ѧϰ�γ̱���\minist\train-images.idx3-ubyte');
labels = loadMNISTLabels('C:\Users\Anthony\Desktop\201883250055-�̺���-����ѧϰ�γ̱���\minist\train-labels.idx1-ubyte');
images = images';  %����images��labels����
[m, n] = size(labels);
b1 = load('b1_final.mat');
b1_final = b1.b1;
b2 = load('b2_final.mat');
b2_final = b2.b2;
theta = load('theta_final.mat');
theta_final = theta.theta;
w = load('w_final.mat');
w_final = w.w;   %����ѵ���õ�����Ȩ��
for i = 1 : m
    if labels(i,1) == 0
        labels(i,1) = 10;
    end
end      %������Ϊ0��label��Ϊ10��������10��label
images = images(1001:1200,:); %ȡ200��Ϊ���Լ�
labels = labels(1001:1200,:);
label = labels';
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
test_Y = zeros(200,10);
for num = 1 : 200
    current_label = zeros(1,10);
    current_label(1,label(1,num)) = 1;
    X = train_data(num,1:end-1);
    T = sigmoid(X * theta_final + b1_final);  %����ó�����T����Ԫ�ڵ�ֵ
    Y(num,:) = sigmoid(T * w_final + b2_final);    %����ó������Y����Ԫ�ڵ�ֵ
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
