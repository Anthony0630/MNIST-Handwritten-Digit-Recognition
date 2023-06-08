function [sigmoid_val] = sigmoid(x)
    sigmoid_val = 1./(1 + exp(-x));
end