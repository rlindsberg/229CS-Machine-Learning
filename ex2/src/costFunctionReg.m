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


% hθ(x) = g(θ.transpose * x) = g(z)
% X is 100 x 3, theta is 3 x 1
z = X * theta;
gOfz = sigmoid(z);

costSum = sum( -y .* log(gOfz) - (1 - y) .* log(1 - gOfz) );
regSum = sum( theta(2:length(theta)).^2 )
J = costSum ./ m + ( lambda / (2 * m) * regSum ) ;

% partiel dir
% (gOfz - y) is 100 x 1
partSum = sum(X .* (gOfz - y) );
grad = partSum ./ m;

% regularization from θ(1), all rows, from column 2
grad(: ,2:length(grad)) = grad(: ,2:length(grad)) + ( lambda / m ) * theta( 2:length(theta) )';


% =============================================================

end