l1_reg(d, W)

% Add referente and comment about the optimization 
% method used
%
%
% ||W*x - d ||^2 - lambda*|x|
% 

Q = lambda*W'*W*eye(size(W,2));
epis = 1e4;

for it = 1:max_it
  g = W'*d;
  x = pinv(W'*W + Q)*g;
  Q = lambda*(1/(abs(x) + epis));
end

mse_error(it) = mean((d-W*x).^2);
