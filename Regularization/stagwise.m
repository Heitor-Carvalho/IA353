function [beta_hist, beta_sum] = stagwise(W, d, epislon, max_it)
% [beta_hist, beta_sum] = stagwise(W, d, epislon, max_it) - Stagwise
% Forward selection, moves a step with length epislon in the direction
% of the most correlated variable. If episolon equals 1, this method
% became the classical FowardSelection method
%
% Inputs:
%  W - Data matrix, each collumn represents a variable and
% each line a sample of this variable set
%  d - Regreesion target valeu 
%
% Outputs: 
%  beta_hist - beta values progression
%  beta_sum  - sum of beta coefitients
%

  % Normalizing variables
  [W, d] = variables_normalize(W, d);
  
  mu = zeros(size(W,1), 1);
  beta = zeros(size(W,2), 1);
  selected_var_idx = logical(zeros(size(W,2), 1));
  beta_sum = zeros(max_it, 1);
  beta_hist = zeros(size(W,2), max_it);
  
  beta_sum(1) = 0;
  beta_hist(:, 1) = beta;
  for i = 2:max_it
    
    % Calculating residual
    residual = d - mu;
    
    % Residual variables correlations
    correlations = W'*residual;
    [max_corr, max_corr_idx] = max(abs(correlations));  % max_corr = C;

    
    % Active set
    selected_var_idx(max_corr_idx) = 1;
    Xa = W(:, selected_var_idx);

    % Calculating equiangular vector (vector equiangular with the collumns of W)
    dk = pinv(Xa'*Xa)*Xa'*residual;

    beta(selected_var_idx) = beta(selected_var_idx) + epislon*dk;
    mu = mu + epislon*Xa*dk;

    beta_hist(:, i) = beta;
    
    beta_sum(i) = sum(abs(beta));
  end

end