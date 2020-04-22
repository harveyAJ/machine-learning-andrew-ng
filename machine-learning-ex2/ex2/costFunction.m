function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
[m, n] = size(X);

% You need to return the following variables correctly 
J = 0;
grad = zeros(n, 1);

% cost function
h_theta = sigmoid(X * theta);
J = -1/m * (y' * log(h_theta) + (1 - y)' * log(1 - h_theta)); 

% derivatives
for j=1:n
  xj = X(:,j);
  grad(j) = 1/m * (h_theta - y)' * xj;   
endfor

% =============================================================

end
