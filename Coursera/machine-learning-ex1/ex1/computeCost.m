function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



%   theta  (2x1)    => convert  [t1  t2] 
%          t1
%          t2
%

%   X     (m x2)          => convert 2 x m  
%         1   x0              1   1   1
%         1   x1              x0   x1  x2
%         1   x2

%  so we get (1 x m) matrix

h =  theta' * X';

D = h - y';
J = (1/(2*m)) * sum(D .^ 2);


% =========================================================================

end
