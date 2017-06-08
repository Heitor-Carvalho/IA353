l1_reg(d, W)

% Add referente and comment about the optimization 
% method used
%
%
% ||W*x - d ||^2 - lambda*|x|
% 

Q = lambda*trace(W'*W)*eye(size(W,2))
epis = 1e-4;

g = W'*d;
for it = 1:max_it
  x = pinv(W'*W + Q)*g;
  Q = 2*mu^2*diag(1./(abs(r).*(sqrt(r.^2 + 4*mu^2) + abs(r))+sc));
end

mse_error(it) = mean((d-W*x).^2);
