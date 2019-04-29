function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th
%               column of X contains the values of X to the p-th power.
%
%



% when a training set X of size m × 1 is passed into the function, the function should return a m×p matrix X_poly
% Now we have mapped features to a higher dimension.
% Part 6 of ex5.m will apply it to the training set, the test set, and the cross validation set (haven’t been used yet).
for j = 1:p
    X_poly(:, j) = X .^ j;
end



% =========================================================================

end
