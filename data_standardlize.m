function [data] = data_standardlize(train_data)
    [s, t] = size(train_data); %����s��������t-1������ά�ȣ����һ��Ϊlabel
    %���б�׼������
    Means = zeros(1,t-1); %��ʼ����ֵ����
    Stds = zeros(1,t-1); %��ʼ����׼�����
    for i = 1 : t-1
        Means(1,i) = mean(train_data(:,i));
        Stds(1,i) = std(train_data(:,i));
    end
    for i = 1 : s
        for j = 1 : t-1
            data(i,j) = (train_data(i,j) - Means(1,j))/Stds(1,j);
        end
    end
    data = [data,train_data(:,end)];
end