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

for i = 1:m
  a1 = X(i,:)';        % Obtain row from X and transpose to column vector
  a1 = [1;a1];         % Add a0 = 1 to input layer
  z2 = Theta1 * a1;    % Compute z2 for 2nd layer
  a2 = sigmoid(z2);    % Compute activation units for 2nd layer
  a2 = [1;a2];         % Add ones to a0 of 2nd layer
  z3 = Theta2 * a2;    % Compute z3
  h = sigmoid(z3);     % Compute probability for each label
  [mx, p(i)] = max(h); % Set prediction to h with highest probability 
endfor

% =========================================================================


end
