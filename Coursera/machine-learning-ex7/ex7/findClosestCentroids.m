function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
% centroids
% X

% You need to return the following variables correctly.

idx = zeros(size(X,1), 1);
K = size(centroids,1);


r = zeros(size(X,1), K);

c = centroids'(:)';
c= repmat(c,size(X,1),1);
x = repmat(X,1,K);
y = (c - x) .^ 2;

irow = size(centroids,1);
icol = size(centroids, 2);

k = 0;
for i=1:irow,
    acc = zeros(size(X,1), 1);
    for j=1:icol,
        k = k + 1;
        acc = acc + y(:,k);
    end
    r(:,i) = acc;
end


% r = [y(:,1)+y(:,2) y(:,3)+y(:,4) y(:,5)+y(:,6)];
[val, idx] = min(r, [], 2);

% for i = 1:size(X,1),
%     for j = 1:K,
%        % r(i,j) = norm(X(i) - centroids(j));
%        r(i,j) = sum((X(i) -centroids(j)) .^ 2);
%    end
% end
% [val, idx] = min(r, [], 2);
% % idx


% ====================== YOUR CODE HERE ====================== 
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

