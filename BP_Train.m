function [theta, b1, b2, w] = BP_Train(train_data, NeuronNum, OutputNum, train_number)
%     train_dataΪѵ�����ݼ�
%     NeuronNumΪ�м�������Ԫ����
%     OutputNumΪ�������Ԫ��������Ԥ��ֵ�ĸ���
%     train_numberΪ������ѵ������
    features = train_data(:,1:end-1); %ѵ������Ϊѵ�����ݳ����һ�������������
    label = train_data(:,end)'; %ѵ�������ǩΪѵ���������һ�ж�Ӧ������
    [m, n] = size(features); %��ȡ���������ĸ�����ά��
    theta = rand(n, NeuronNum)/2; %��ʼ�������X������T�м��thetaȨ�ؾ���
    b1 = rand(1, NeuronNum)/3; %��ʼ������T��b1ƫ�þ���
    w = rand(NeuronNum, OutputNum)/2.5; %��ʼ������T�������Y�м��wȨ�ؾ���
    b2 = rand(1, OutputNum)/2; %��ʼ�������Y��b2ƫ�þ���
    delta = 0.1; %ѧϰ��
    count = 0; %������¼ѵ������
    while count < train_number
        for num = 1 : m
            current_label = zeros(1,OutputNum);
            current_label(1,label(1,num)) = 1;
            X = features(num,:);
            T = sigmoid(X * theta + b1);  %����ó�����T����Ԫ�ڵ�ֵ
            Y = sigmoid(T * w + b2);    %����ó������Y����Ԫ�ڵ�ֵ
            % ��ƫ��
            d_w = zeros(NeuronNum, OutputNum);
            for i = 1 : NeuronNum
                for j = 1 : OutputNum
                    d_w(i,j) = (Y(1,j)-current_label(1,j))*Y(1,j)*(1-Y(1,j))*T(1,i);
                end
            end  %�����w��ƫ������
            d_theta = zeros(n, NeuronNum);
            for Num = 1 : OutputNum
                for i = 1 : n
                    for j = 1 : NeuronNum
                        d_theta(i,j) = d_theta(i,j)+(Y(1,Num)-current_label(1,Num))*Y(1,Num)*(1-Y(1,Num))*w(j,Num)*T(1,j)*(1-T(1,j))*X(1,i);
                    end
                end
            end %�����theta��ƫ������
            d_b2 = zeros(1, OutputNum);
            for i = 1 : OutputNum
                d_b2(1,i) = (Y(1,i)-current_label(1,i))*Y(1,i)*(1-Y(1,i));
            end %�����b2��ƫ������
            d_b1 = zeros(1, NeuronNum);
            for k = 1 : OutputNum
                for j = 1: NeuronNum
                    d_b1(1,j) = d_b1(1,j)+(Y(1,k)-current_label(1,k))*Y(1,k)*(1-Y(1,k))*w(j,k)*T(1,j)*(1-T(1,j));
                end
            end %�����b1��ƫ������
            %���²���
            w = w - delta * d_w;
            theta = theta - delta * d_theta;
            b1 = b1 - delta * d_b1;
            b2 = b2 - delta * d_b2;
        end
        count = count + 1;
    end
            
            