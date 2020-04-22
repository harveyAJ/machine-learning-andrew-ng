function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m, n] = size(X);

% You need to return the following variables correctly 
J = 0;
grad = zeros(n, 1);

% cost function
h_theta = sigmoid(X * theta);
J = -1/m * (y' * log(h_theta) + (1 - y)' * log(1 - h_theta)) + ...
   0.5*lambda/m * (theta'*theta - theta(1)^2); 

% derivatives
grad(1) = 1/m * (h_theta - y)' * X(:,1);
for j=2:n
  xj = X(:,j);
  grad(j) = 1/m * (h_theta - y)' * xj + lambda/m * theta(j);   
endfor

% =============================================================

end
