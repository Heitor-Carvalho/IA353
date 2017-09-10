function [beta_hist, beta_sum, W, d] = lars_lasso(H, t, unorm, max_components)
% [beta_hist, beta_sum, W, d] = lars_lasso(H, t, unorm, max_components) - Least Angle Regression algorithm
% fmodified to find the LASSO solution. 
% This implementation follows the paper 2004 Least Angle Regression 
% paper notation.
%
% Inputs:
%  H - Data matrix, each collumn represents a variable and
% each line a sample of this variable set
%  t - Regreesion target valeu
%  unnorm - If one, return beta in the original data scale, 
% in this case is necessary to add a collumn of ones in the matrix data.  
% max_components - Maximun number of components
%
% Outputs: 
%  W - Normalized data matrix
%  d - Normalized target value
%  beta_hist - beta values progression
%  beta_sum  - sum of beta coefitients
%
  [W, d, avg, var_energy] = variables_normalize(H, t);
  
  mu = zeros(size(W,1), 1);
  beta = zeros(size(W,2), 1);
  selected_var_idx = logical(zeros(size(W,2), 1));
  beta_sum = zeros(size(W,2), 1);
  beta_hist = zeros(size(W,2), size(W,2));
  
  beta_sum(1) = 0;
  beta_hist(:, 1) = beta;
  residual = d - mu;
  correlations = W'*residual;
  correlations_sign = sign(correlations);
  [max_corr, max_corr_idx] = max(abs(correlations));  % max_corr = C;
  selected_var_idx(find(abs(correlations) == max_corr)) = 1;
  end_lasso = 1;

  i = 2;
  while(end_lasso && sum(selected_var_idx) <= max_components) 
    residual = d - mu;
    correlations = W'*residual;
    correlations_sign = sign(correlations);
    [max_corr, max_corr_idx] = max(abs(correlations));  % max_corr = C;
    Xa = W(:, selected_var_idx).*repmat(correlations_sign(selected_var_idx)', size(W,1), 1);

    % Calculating equiangular vector (vector equiangular with the collumns of W)
    G = Xa'*Xa;
    Ginv = pinv(G);
    A = 1/sqrt(sum(Ginv(:)));
    w = A*sum(Ginv,2);
    u = Xa*w;

    % Correlation of inputs with the equiangular vector
    a = W'*u;
   
    gamma_minus = (max_corr - correlations(~selected_var_idx))./(A-a(~selected_var_idx));
    gamma_minus(gamma_minus <= 0) = max_corr + 1;
    gamma_plus = (max_corr + correlations(~selected_var_idx))./(A+a(~selected_var_idx));
    gamma_plus(gamma_plus <= 0) = max_corr + 1;
    [gamma_candidates, gamma_candidates_idx] =   min([[gamma_minus; max_corr/A], [gamma_plus; max_corr/A]]);
    [gamma, gamma_idx]  = min(gamma_candidates);

    % Checking LASSO condition
    gamma_ = -beta(selected_var_idx)./(w.*correlations_sign(selected_var_idx));
    gamma_(gamma_ <= 0) = gamma + 1;
    [min_gamma, min_gamma_idx] = min(gamma_);

    current_var_pos = find(selected_var_idx);
    complement_var_pos = find(~selected_var_idx);
    if(min_gamma < gamma)
      gamma = min_gamma;
      beta(selected_var_idx) = beta(selected_var_idx) + gamma*w.*correlations_sign(selected_var_idx);
      beta(current_var_pos(min_gamma_idx)) = 0;
      selected_var_idx(current_var_pos(min_gamma_idx)) = 0;
    else
      beta(selected_var_idx) = beta(selected_var_idx) + gamma*w.*correlations_sign(selected_var_idx);
      if(~isempty(complement_var_pos))
        selected_var_idx(complement_var_pos(gamma_candidates_idx(gamma_idx)))  = 1;
      else
        % Trying to add more than size(W, 2) variables
        end_lasso = 0;
      end
    end 
    mu = mu + gamma*u;
    
    beta_hist(:, i) = beta;
    beta_sum(i) = sum(abs(beta));

    i = i + 1;
  end
  
  if(unorm)
    beta_h_scaled = beta_hist./repmat(sqrt(var_energy'), 1, size(beta_hist, 2));
    beta_hist = [mean(t)-avg*beta_h_scaled; beta_h_scaled];
    beta_sum = sum(beta_hist, 1);
  end 

end
