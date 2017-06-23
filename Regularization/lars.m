function [beta_hist, beta_sum, W, d] = lars(H, t)
% [beta_hist, beta_sum] = lars(H, t) - Least Angle Regression algorithm
% find the regression coeficients moving in the least angle
% direction. The variables became equally correlated with the residual
% at each step.
% This implementation follows the paper 2004 Least Angle Regression 
% paper notation.
%
% Inputs:
%  H - Data matrix, each collumn represents a variable and
% each line a sample of this variable set
%  t - Regreesion target valeu 
%
% Outputs: 
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
  for i = 2:size(W,2)+1

    residual = d - mu;
    correlations = W'*residual;
    correlations_sign = sign(correlations);
    [max_corr, max_corr_idx] = max(abs(correlations));  % max_corr = C;
    selected_var_idx(abs(max_corr - abs(correlations)) < 1e-9) = 1;
    Xa = W(:, selected_var_idx).*repmat(correlations_sign(selected_var_idx)', size(W,1), 1);

    % Calculating equiangular vector (vector equiangular with the collumns of W)
    G = Xa'*Xa;
    Ginv = inv(G);
    A = 1/sqrt(sum(Ginv(:)));
    w = A*sum(Ginv,2);
    u = Xa*w;

    % Correlation of inputs with the equiangular vector
    a = W'*u;

    gamma_minus = (max_corr - correlations(~selected_var_idx))./(A-a(~selected_var_idx));
    gamma_minus(gamma_minus < 0) = max_corr + 1;
    gamma_plus = (max_corr + correlations(~selected_var_idx))./(A+a(~selected_var_idx));
    gamma_plus(gamma_plus < 0) = max_corr + 1;
    [gamma_candidates, gamma_candidates_idx] =   min([gamma_minus, gamma_plus]);
    [gamma, gamma_idx]  = min(gamma_candidates);
    if(isempty(gamma))
      gamma = max_corr/A;
    end
    mu = mu + gamma*u;

    beta(selected_var_idx) = beta(selected_var_idx) + gamma*w.*correlations_sign(selected_var_idx);
    beta_hist(:, i) = beta;
    beta_sum(i) = sum(abs(beta));
  end
end
