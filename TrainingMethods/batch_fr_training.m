function [nn_, err_hist, it] = batch_fr_training(train_set, target, nn, train_par)
  % To do: add support to multiples outputs
  % To do: add function description

  nn_ = nn;
  mse_error = train_par.max_error;
  err_hist = zeros(1, train_par.max_it);
  it = 0;

  train_par.beta = 0;

  % Calculate gradient using backpropagation
  g_i = -back_prop_batch_gradient(train_set, target, nn_);
  d = g_i; 
  samples_sz = size(train_set, 2);
  
  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
    % Get weiths from neuro network structure    
    weigths = convert_neuronet_vw_to_w(nn_);
    
    % Functional to be minimized
    Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths + alpha*d, nn_))).^2);

    % Line search for alpha
    train_par.alpha = golden_search(0, 10, Jfunc, 1e-3)

    % Training method Fletcher-Reeves
    weigths = weigths + train_par.alpha*d;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);
    
    g_i1 = -back_prop_batch_gradient(train_set, target, nn_);
    
    beta = max(0, (g_i1'*g_i)/(g_i'*g_i));

    d = g_i1 + beta*d;

    % Calculation MSE error
    mse_error = mean((target - neural_nete(train_set, nn_)).^2);

    it = it + 1;
    g_i1 = g_i;
    err_hist(it) = mse_error;
   
  end
  
end