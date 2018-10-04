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

% Implement h
% size(Theta1) % -> 25 x 401
% size(Theta2) %  -> 10 x 26
% size(X) % -> 5000 x 401
X_orig = X;
% size(X_orig) %-> 5000 x 400

X = [ones(m, 1) X];
% size(X) %-> 5000 x 401
z2 = Theta1 * X';
% size(z2) %-> 25 x 5000
a2 = sigmoid(z2);
% size(a2) % ->  25 x 5000
m1 = size(a2, 2);

a2_orig = a2;
% size(a2_orig) % -> 25 x 5000

a2 = [ones(1, m1); a2];
% size(a2) % -> 26 x 5000

z3 = Theta2 * a2; 
% size(z3) % -> 10 x 5000

h = sigmoid(z3);
% size(h) % -> 10 x 5000

% size(y) % 5000 x 1
% recode y
% y_vec = zeros(num_labels, m);
% size(y_vec) % 10 x 5000

labels = (1:num_labels)';
y_vec = repmat(labels, [1, m]);
% size(y_vec) % 10 x 5000
y_vec = (y_vec == y');
% size(y_vec) % 10 x 5000

 
p = (log(h) .* y_vec .* (-1)) - ((1 - y_vec) .* log(1 - h));
% size(p) % 10 x 5000
s1 = sum(p,2);

% size(s1) % 10 x 1

J = ((1/m) * sum(s1));

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

   % d3 = a3 - y_vec
   d3 = h - y_vec;
  
   % size(d3) % 10x5000
   % size(Theta2) %  -> 10 x 26
   
   %size(z2) % 10 x 5000;

   t = (Theta2' * d3);
   t(1,:) = []; % remove the row 1, corresponding to bias term.
   % size(t) % 25 x 5000

   d2 =  t .*  sigmoidGradient(z2);


   % size(d3) % 10 x 5000
   % size(d2) % 25 x 5000

   %D2
   D2  = d3 * a2'; 
   % size(D2) % 10 x 26

   % D1
   D1 =  d2 * X;
   % size(D1) % 25 x 401
   Theta1_grad = D1/m;
   Theta2_grad = D2/m;


% size(Theta1) % -> 25 x 401
% size(Theta2) %  -> 10 x 26

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_r = Theta1;
Theta2_r = Theta2;

Theta1_r(:,1) = 0;
Theta2_r(:,1) = 0;

 J = J + (lambda * (sum(sum(Theta1_r .^ 2)) + sum(sum(Theta2_r .^ 2)))/(2*m));

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
