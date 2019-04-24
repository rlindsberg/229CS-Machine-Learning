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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Add bias to the X data matrix
X = [ones(m, 1) X];
% layer 1
a_superscript_1 = X;

% layer 2, hidden
z_superscript_2 = X * Theta1';
a_superscript_2 = sigmoid(z_superscript_2);
a_superscript_2 = [ones(size(a_superscript_2, 1), 1) a_superscript_2];

% layer 3
z_superscript_3 = a_superscript_2 * Theta2';
a_superscript_3 = sigmoid(z_superscript_3);
% output
h_theta = a_superscript_3;

% remap y (5000x1 vector) to y (10x5000 matrix)
% init y_10x5000
y_10x5000 = zeros(num_labels, m); % m = size(X, 1) = size of the training set

for i=1:m
    % y(i) is the label, set 1 at the corresponding index in the i th column
    y_10x5000( y(i), i) = 1;
end

% compute cost
% sum(matrix) sums up all rows

costSum = sum(
            sum( -y_10x5000' .* log(h_theta) - (1 - y_10x5000') .* log(1 - h_theta) )
          );

% Regularization for a 3-layer-nn
regSum = sum(sum( Theta1(:,2:end).^2 )) + sum(sum( Theta2(:,2:end).^2 ));
fprintf('reg cost is: ')
lambda / (2 * m) * regSum
J = 1/m * costSum + ( lambda / (2 * m) * regSum );


% Backpropagation
for t=1:m
    % step 1, feedforward
    x = X(t, :);
    % % Add bias to the X data matrix
    % x = [1 x]; % another way x = [1; X(t,:)'];
    % layer 1
    a_superscript_1 = x;

    % layer 2, hidden
    z_superscript_2 = a_superscript_1 * Theta1'; % size is 1x25
    a_superscript_2 = sigmoid(z_superscript_2); % size is 1x25
    a_superscript_2 = [1 a_superscript_2]; % size is 1x26

    % layer 3
    z_superscript_3 = a_superscript_2 * Theta2'; % 1x26 * 26x10 = 1x10
    a_superscript_3 = sigmoid(z_superscript_3);
    % output
    h_theta = a_superscript_3;

    % step 2, calculate δ
    % create labels [1:10] (i.e. 1    2    3    4    5    6    7    8    9   10)
    y_subscript_k = [1 : num_labels] == y(t); % t th data
    delta_superscript_3 = a_superscript_3 - y_subscript_k;
end




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
