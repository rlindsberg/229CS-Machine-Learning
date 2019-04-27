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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Cost
h_subscript_theta = X * theta;
costSum = sum( (h_subscript_theta - y).^2 );
regSum = sum( theta(2:end).^2 ); % do not regularize theta_subscript_0, i.e. theta(1)

J = 1/(2 * m) * costSum + ( lambda / (2 * m) * regSum );


% Gradient
partSum = sum(X .* (h_subscript_theta - y) );
grad = 1/m * partSum;

% regularization from θ(1), all rows, from column 2
grad(: ,2:length(grad)) = grad(: ,2:length(grad)) + ( lambda / m ) * theta( 2:length(theta) )';



% =========================================================================

grad = grad(:);

end
