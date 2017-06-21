function [beta_hist, beta_sum] = lars(W, d, max_it)
% [beta_hist, beta_sum] = lars(W, d) - Least Angle Regression algorithm
% find the regression coeficients moving in the least angle
% direction. The variables became equally correlated with the residual
% at each step.
%
% Inputs:
%  W - Data matrix, each collumn represents a variable and
% each line a sample of this variable set
%  d - Regreesion target valeu 
%  max_it - Algorithm maximum iteration

  % Normalizing variables
  [W, d] = variables_normalize(W, d);
  
  mu = zeros(size(W,1), 1);
  beta = zeros(size(W,2), 1);
  selected_var_idx = logical(zeros(size(W,2), 1));
  beta_sum = zeros(max_it, 1);
  beta_hist = zeros(size(W,2), max_it);
  
  beta_sum(1) = 0;
  beta_hist(:, 1) = beta;
  change_var = [];
  i = 2;
  while(i < max_it)
 
    % Calculating residual
    residual = d - mu;
    
    % Residual variables correlations
    correlations = W'*residual;
    [max_corr, max_corr_idx] = max(abs(correlations));  % max_corr = C;
    selected_var_idx(abs(max_corr - abs(correlations)) < 1e-5) = 1;
    selected_var_idx(change_var) = 0;

    % Active set
    [dk, Xa] = equiangular_active_set(W, selected_var_idx, residual);        
    [gamma] = get_gamma_coef(max_corr, correlations, selected_var_idx, dk, Xa, W);
    
    beta_new = beta;
    beta_new(selected_var_idx) = beta(selected_var_idx) + gamma*dk;
    
    change_var = find(beta.*beta_new < 0);
    
    if(~isempty(change_var))
      change_var_dk = find(beta(selected_var_idx).*beta_new(selected_var_idx) < 0);
      gamma = -beta(change_var)./dk(change_var_dk);
      [gamma, min_gamma_idx] = min(gamma);
      change_var = change_var(min_gamma_idx);
    else
      change_var = [];
    end
    
    beta(selected_var_idx) = beta(selected_var_idx) + gamma*dk;
    mu = mu + gamma*Xa*dk;
    beta_hist(:, i) = beta;
    
    beta_sum(i) = sum(abs(beta));
    if(abs(gamma - 1) < 1e-2)
      beta_hist(:, i+1:end) = [];
      beta_sum(i+1:end) = [];
      break
    end
    
    i = i + 1;
  end

end

function [gamma] = get_gamma_coef(max_corr, correlations, selected_var_idx, dk, Xa, W)

    % Correlation between OLS estimation with active set and the variables
    a = W'*Xa*dk;
    A = max(a);    

    %%% Remiver o changer dos competidores para ficar com correlação
    %%% cruzada iguais
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

end

function [dk, Xa] = equiangular_active_set(W, selected_var_idx, residual)
    % Active set
    Xa = W(:, selected_var_idx);
    
    % Calculating equiangular vector (vector equiangular with the collumns of W)
    dk = pinv(Xa'*Xa)*Xa'*residual;
end