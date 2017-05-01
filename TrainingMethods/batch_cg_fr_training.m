function [nn_, err_hist, it] = batch_cg_fr_training(input_sets, targets, nn, train_par)
  % To do: add support to multiples outputs
  % To do: add function description

  % Getting training and target sets:
  [train_set, input_sets] = get_train_set(input_sets);
  [target, targets] = get_train_target(targets);

  nn_ = nn;

  mse_error = train_par.max_error;
  err_hist = zeros(size(input_sets, 2), train_par.max_it);
  it = 0;

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
    alpha = golden_search(0, 1, Jfunc, 1e-3);

    % Training method Fletcher-Reeves
    weigths = weigths + alpha*d;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);
    
    g_i1 = -back_prop_batch_gradient(train_set, target, nn_);
    
    beta = max(0, (g_i1'*g_i)/(g_i'*g_i));

    if(mod(it, length(g_i)) == 0)
      d = g_i1;
    else
      d = g_i1 + beta*d;
    end

    % Calculation MSE error
    err_hist(:, it+1) = get_mse_error(input_sets, targets, nn_);

    mse_error = err_hist(1, it+1);
    it = it + 1;
    g_i1 = g_i;
   
  end
  
end