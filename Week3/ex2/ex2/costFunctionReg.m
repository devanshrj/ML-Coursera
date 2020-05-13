function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(theta,1);

h = X * theta;

temp = y .* log( sigmoid(h) ) + ( 1 - y ) .* log( 1 - sigmoid(h) );

J = (-1 / m) * sum( sum(temp) ) + (lambda / (2 * m) * sum( theta(2:n,:).^2) );

for i = 1:n
    if (i == 1)
        grad(i) = (1 / m) * sum ( ( sigmoid(h) - y ) .* X(:,i) );

    else
        grad(i) = (1 / m) * sum ( ( sigmoid(h) - y ) .* X(:,i) ) + ( lambda / m * theta(i) );

    endif
endfor

% =============================================================

end
