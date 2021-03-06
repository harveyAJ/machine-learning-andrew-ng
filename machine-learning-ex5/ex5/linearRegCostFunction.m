function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

error = X * theta - y;

J = 0.5 / m * error' * error + ...
    lambda * 0.5 / m * (theta' * theta - theta(1)^2);

grad = 1/m * X' * error;

grad(2:end) += lambda/m * theta(2:end);

% =========================================================================

grad = grad(:);

end
