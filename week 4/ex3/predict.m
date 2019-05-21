function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m,1) X];
a2 = sigmoid(X*Theta1.');
%ith row of a2 contains the 2nd layer elements for test case of ith row of X 
a2 = [ones(m,1) a2];
a3 = sigmoid(a2*Theta2.');
%ith row of a3 contain num_labels number of elements corresponding to the probabilities of being in that label for test case of ith row of X 

[~,label] = max(a3, [], 2);
p = label;
% =========================================================================
end

function g = sigmoid(z)
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[r,c] = size(z);
for in = 1:r
    for j = 1:c
        g(in,j) = 1/(1+exp(-z(in,j)));
    end
end
% =============================================================
end




