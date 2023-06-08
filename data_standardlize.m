function [data] = data_standardlize(train_data)
    [s, t] = size(train_data); %共有s个样本，t-1个特征维度，最后一列为label
    %进行标准化处理
    Means = zeros(1,t-1); %初始化均值矩阵
    Stds = zeros(1,t-1); %初始化标准差矩阵
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