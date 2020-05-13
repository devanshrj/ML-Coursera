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
temp = 0;

% feedforward propogation implementation.
% every row in X is a training example. Taking training example 'c', find a2 and a3 by using given
% weights. max returns highest probability to temp and index of highest probability, i.e., required
% label to p(c)

for c = 1:m

    z2 = Theta1 * X(c,:)';
    a2 = [ 1; sigmoid(z2)];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    [temp p(c)] = max(a3);

end


% =========================================================================


end
