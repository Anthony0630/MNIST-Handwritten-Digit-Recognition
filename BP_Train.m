function [theta, b1, b2, w] = BP_Train(train_data, NeuronNum, OutputNum, train_number)
%     train_data为训练数据集
%     NeuronNum为中间隐层神经元个数
%     OutputNum为输出层神经元个数，即预期值的个数
%     train_number为神经网络训练次数
    features = train_data(:,1:end-1); %训练特征为训练数据除最后一列外的所有数据
    label = train_data(:,end)'; %训练结果标签为训练数据最后一列对应的数据
    [m, n] = size(features); %获取特征向量的个数和维数
    theta = rand(n, NeuronNum)/2; %初始化输入层X与隐层T中间的theta权重矩阵
    b1 = rand(1, NeuronNum)/3; %初始化隐层T的b1偏置矩阵
    w = rand(NeuronNum, OutputNum)/2.5; %初始化隐层T与输出层Y中间的w权重矩阵
    b2 = rand(1, OutputNum)/2; %初始化输出层Y的b2偏置矩阵
    delta = 0.1; %学习率
    count = 0; %用来记录训练次数
    while count < train_number
        for num = 1 : m
            current_label = zeros(1,OutputNum);
            current_label(1,label(1,num)) = 1;
            X = features(num,:);
            T = sigmoid(X * theta + b1);  %计算得出隐层T的神经元节点值
            Y = sigmoid(T * w + b2);    %计算得出输出层Y的神经元节点值
            % 求偏导
            d_w = zeros(NeuronNum, OutputNum);
            for i = 1 : NeuronNum
                for j = 1 : OutputNum
                    d_w(i,j) = (Y(1,j)-current_label(1,j))*Y(1,j)*(1-Y(1,j))*T(1,i);
                end
            end  %求出对w的偏导矩阵
            d_theta = zeros(n, NeuronNum);
            for Num = 1 : OutputNum
                for i = 1 : n
                    for j = 1 : NeuronNum
                        d_theta(i,j) = d_theta(i,j)+(Y(1,Num)-current_label(1,Num))*Y(1,Num)*(1-Y(1,Num))*w(j,Num)*T(1,j)*(1-T(1,j))*X(1,i);
                    end
                end
            end %求出对theta的偏导矩阵
            d_b2 = zeros(1, OutputNum);
            for i = 1 : OutputNum
                d_b2(1,i) = (Y(1,i)-current_label(1,i))*Y(1,i)*(1-Y(1,i));
            end %求出对b2的偏导矩阵
            d_b1 = zeros(1, NeuronNum);
            for k = 1 : OutputNum
                for j = 1: NeuronNum
                    d_b1(1,j) = d_b1(1,j)+(Y(1,k)-current_label(1,k))*Y(1,k)*(1-Y(1,k))*w(j,k)*T(1,j)*(1-T(1,j));
                end
            end %求出对b1的偏导矩阵
            %更新参数
            w = w - delta * d_w;
            theta = theta - delta * d_theta;
            b1 = b1 - delta * d_b1;
            b2 = b2 - delta * d_b2;
        end
        count = count + 1;
    end
            
            