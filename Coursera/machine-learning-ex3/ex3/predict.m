function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% size(Theta1)  -> 25 x 401
% size(Theta2)  -> 10 x 26
% size(X) -> 5000 x 400
X = [ones(m, 1) X];
% size(X)-> 5000 x 401
z1 = Theta1 * X';
% size(z1) -> 25 x 5000
a2 = sigmoid(z1);
% size(a2) ->  25 x 5000
m1 = size(a2, 2);

a2 = [ones(1, m1); a2];

% size(a2) -> 26 x 5000

z2 = Theta2 * a2;

%size(z2) -> 10 x 5000

h = sigmoid(z2);
% size(h) -> 10 x 5000

[i, p] = max(h, [], 1);
size(p)
p = p';








% =========================================================================


end
