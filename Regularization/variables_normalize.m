function [Wnorm, dnorm, avg, var_energy] = variables_normalize(W, d)
% [Wnorm, dnorm] = variables_normalize(W, d) - Normalize variables
% variables in W with energy one and remove the average value from d
% Inputs:
%  W - Data matrix, each collumn represents a variable and
% each line a sample of this variable set
%  d - Regreesion target valeu 

  % Normalizing data
  avg = mean(W,1);
  W = W - repmat(avg, size(W,1), 1);
  var_energy = sum(W.^2);
  Wnorm = W./repmat(sqrt(var_energy), size(W,1), 1);
  dnorm = d - mean(d);

end
