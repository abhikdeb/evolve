function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    R1 = 0;
    R2 = 0;
    for i = 1:m,
        y1 = 0;
        for j = 1:length(theta),
          y1 += theta(j)*X(i,j);
        end;
        R1 += (y1 - y(i))*X(i,1);
        R2 += (y1 - y(i))*X(i,2);
      end;
    theta(1) -= ((alpha/m)*R1);
    theta(2) -= ((alpha/m)*R2);
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
