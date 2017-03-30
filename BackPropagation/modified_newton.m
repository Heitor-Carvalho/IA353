function [weigths] = modified_newton(J, diffJ, weigths, train, mse_error)
  
 diff2J = 2*pinv(J*J' + train_par.mu*eye(size(J,1)));
 weigths = weigths - train_par.alha*d2J*J*mse_error;
  
end