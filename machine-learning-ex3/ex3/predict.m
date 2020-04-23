function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

A2 = sigmoid(Theta1 * X'); % (n_hidden_layers,n+1) * (n+1,m)
A2 = [ones(1, m); A2];

[max_values, p] = max(sigmoid(Theta2 * A2), [], 1);

p = p';

% =========================================================================

end
