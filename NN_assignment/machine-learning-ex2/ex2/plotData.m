function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% plot (x, y, "or", x, y2, x, y3, "m", x, y4, "+")



i1 = find(y(:,1) == 1);
o1 = find(y(:,1) == 0);

plot(X(i1,1), X(i1,2), 'xb', "markersize", 10, "linewidth", 2);

plot(X(o1,1), X(o1,2), 'og', "markersize", 10, "linewidth", 2);







% =========================================================================



hold off;

end
