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

% feedforward propagation
X = [ones(m, 1), X]; % 5000x401
a_super_2 = sigmoid(Theta1 * X'); % 25x5000=(25x401)*(401x5000)
a_super_2 = [ones(1, m); a_super_2]; % 26x5000
a_super_3 = sigmoid(Theta2 * a_super_2); % 10x5000=(10x26)*(26*5000)

% non-regularized cost function
I = eye(num_labels); % 10x10
y = I(:,y); % y:5000x1 -> 10x5000
tempJ = (y .* log(a_super_3) - (y - 1) .* log(1 - a_super_3)) / (-m);
J = sum(tempJ(:));

% regularized cost function
tempTheta1 = Theta1.^2;
tempTheta1 = tempTheta1(:,2:end);
tempTheta2 = Theta2.^2;
tempTheta2 = tempTheta2(:,2:end);
J = J + lambda / (2 * m) * (sum(tempTheta1(:)) + sum(tempTheta2(:)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta_super_3 = a_super_3 - y; % 10x5000
delta_super_2 = Theta2(:,2:end)' * delta_super_3 .* sigmoidGradient(Theta1 * X'); % 25x5000

Delta_super_2 = delta_super_3 * a_super_2'; % 10x26=(10x5000)*(5000x26)
Delta_super_1 = delta_super_2 * X; % 25x401=(25x5000)*(5000x401)

% gradient for non-regularized BP
Theta1_grad = Delta_super_1 / m;
Theta2_grad = Delta_super_2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% regularized gradients
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad + lambda / m * Theta1;
Theta2_grad = Theta2_grad + lambda / m * Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
