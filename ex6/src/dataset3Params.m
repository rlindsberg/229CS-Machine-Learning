function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%



% results is 64x3. 64 st. combinations and 3 columns: C, sigma, err
results = zeros(64,3);
C = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma = [0.01 0.03 0.1 0.3 1 3 10 30];

for i = 1:length(C)
    C_subscript_i = C(i);
    sprintf('\ntesting C_subscript_i: ');
    C_subscript_i

    for j = 1:length(sigma)
        sigma_subscript_j = sigma(j);
        sprintf('\ntesting sigma_subscript_i: ');
        sigma_subscript_j

        % [model] = svmTrain(X, Y, C, kernelFunction, tol, max_passes)
        kernelFunction = @(x1, x2) gaussianKernel(x1, x2, sigma_subscript_j);
        model = svmTrain(X, y, C_subscript_i, kernelFunction);
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));

        % writes to the corresponding row in results matrix
        results((i * j + 1), : ) = [C_subscript_i, sigma_subscript_j, prediction_error]

    end
end

% =========================================================================

end
