function J = costFunction(X, theta, y)
m = size(X, 1);
temp = X * theta;
errorsSquared = (temp - y) .^ 2;
J = 1 / (2 * m) * sum(errorsSquared);
