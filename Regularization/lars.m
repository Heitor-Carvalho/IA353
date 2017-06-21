function [beta_hist, beta_sum] = lars(W, d)
% [beta_hist, beta_sum] = lars(W, d) - Least Angle Regression algorithm
% find the regression coeficients moving in the least angle
% direction. The variables became equally correlated with the residual
% at each step.
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
  beta_sum = zeros(size(W,2)+1, 1);
  beta_hist = zeros(size(W,2), size(W,2)+1);
  
  beta_sum(1) = 0;
  beta_hist(:, 1) = beta;
  for i = 2:size(W,2)+1
    
    % Calculating residual
    residual = d - mu;
    
    % Residual variables correlations
    correlations = W'*residual;
    [max_corr, max_corr_idx] = max(abs(correlations));  % max_corr = C;

    
    % Active set
    selected_var_idx(abs(max_corr - abs(correlations)) < 1e-5) = 1;
    Xa = W(:, selected_var_idx);
    
    % Calculating equiangular vector (vector equiangular with the collumns of W)
    dk = pinv(Xa'*Xa)*Xa'*residual;
    
    % Correlation between OLS estimation with active set and the variables
    a = W'*Xa*dk;
    A = max(a);    

    % Correlation of inputs with the equiangular vector
    gamma_minus = (max_corr - correlations(~selected_var_idx))./(A-a(~selected_var_idx));
    gamma_minus(gamma_minus < 0) = max_corr + 1;
    gamma_plus = (max_corr + correlations(~selected_var_idx))./(A+a(~selected_var_idx));
    gamma_plus(gamma_plus < 0) = max_corr + 1;
    [gamma_candidates, gamma_candidates_idx] = min([gamma_minus, gamma_plus]);
    [gamma, gamma_idx]  = min(gamma_candidates);
    if(isempty(gamma))
      gamma = max_corr/A;
    end
    
    beta(selected_var_idx) = beta(selected_var_idx) + gamma*dk;

    mu = mu + gamma*Xa*dk;
    beta_hist(:, i) = beta;
    beta_sum(i) = sum(abs(beta));
  end

end