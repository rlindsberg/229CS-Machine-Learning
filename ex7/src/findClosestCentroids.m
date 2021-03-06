function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%



shortestPath = 0;

for i = 1:size(X,1)

    sprintf('Testing %d', i)

    shortestPathSquared = sum( (X(i,:) - centroids(1,:)).^2 );
    shortestPath = sqrt(shortestPathSquared);
    idx(i,1) = 1;
    sprintf('shortestPath is %f\n', shortestPath)

    for j = 2:K
        sprintf('\n Inner loop begins\n')
        distanceSquared = sum( (X(i,:) - centroids(j,:)).^2 );
        sprintf('The X coordinates are %d, %d', X(i), X(1))

        distance = sqrt(distanceSquared);
        sprintf('The distance between X(%d, 1) and centroids(%d, 2) is %f', i, j, distance)
        if distance < shortestPath
            sprintf('\n******** Found a shorter path!!!\n')
            shortestPath = distance
            sprintf('********\n\n')
            idx(i,1) = j;
        endif
    end
end




% =============================================================

end
