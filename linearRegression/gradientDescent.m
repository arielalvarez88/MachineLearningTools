function [theta, J_history] = gradientDescent(X,y,theta,alpha,num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
    m = length(y);
    J_history = zeros(num_iters,1);


    for i = 1:num_iters
        dif = (theta'*X' - y');
        change = (dif * X)*alpha/m;
        theta = theta - change';                
        error = sum(dif .^ 2)/(2*m);        
        J_history(i,1) = error;        
    end

end