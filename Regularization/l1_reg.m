function [x, mse_error] = l1_reg(d, W, lambda, max_it)
% L1 norm regularization, solve x that minimizes the equation 
%
% ||W*x - d ||^2 - lambda*|x|
% 
% Inputs:
%   d         : target samples
%   w         : pattern matrix
%   lambda    : regularization parameter
%   max_it    : max iteration number
%
% Outputs:
%   x         : solved x
%   mse_error : reconstruction error ||W*x - d ||^2
%
  Q = lambda*trace(W'*W)*eye(size(W,2));
  epis = 1e-7;

  g = W'*d;
  for it = 1:max_it
    x = pinv(W'*W + Q)*g;
    Q = lambda*diag((1./(abs(x) + epis)));
  end

  mse_error = mean((d-W*x).^2);

end