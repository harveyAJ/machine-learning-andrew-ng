function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add first column of ones
%X = [ones(m,1) X];

for i = 1:m
  y_i = zeros(num_labels, 1);
  y_i(y(i)) = 1; %vector of dimension K (=num_labels)
  
  % Forward propagation
  a_1 = X(i,:)'; 
  a_1 = [1; a_1]; % 401 1
  z_2 = Theta1 * a_1; %25 1
  a_2 = sigmoid(z_2); %25 1
  a_2 = [1; a_2]; %26 1
  z_3 = Theta2 * a_2; %10 1
  a_3 = sigmoid(z_3); %10 1
  
  J += y_i' * log(a_3) + (1 - y_i)'*log(1 - a_3);
  
  delta_3 = a_3 - y_i; %10 1
  delta_2 = Theta2' * delta_3;
  delta_2 = delta_2(2:end);
  delta_2 = delta_2 .* sigmoidGradient(z_2); %25 1
  
  Theta1_grad += delta_2 * a_1'; %25 401
  Theta2_grad += delta_3 * a_2'; %10 26
endfor

% Add regularization terms
reg = sum(sum(Theta1(:,2:end) .* Theta1(:,2:end))) + ...
      sum(sum(Theta2(:,2:end) .* Theta2(:,2:end)));

J += -0.5 * lambda * reg; 
 
J /= -m;

Reg1 = lambda * [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Reg2 = lambda * [zeros(num_labels,1) Theta2(:,2:end)];

Theta1_grad += Reg1;
Theta2_grad += Reg2;

Theta1_grad /= m;
Theta2_grad /= m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
