function [nn_, err_hist, it] = batch_oss_training(input_sets, targets, nn, train_par)
  % To do: add support to multiples outputs
  % To do: add function description

  % Getting training and target sets:
  [train_set, input_sets] = get_train_set(input_sets);
  [target, targets] = get_train_target(targets);

  nn_ = nn;

  mse_error = train_par.max_error;
  err_hist = zeros(size(input_sets, 2), train_par.max_it);
  it = 0;

  samples_sz = size(train_set, 2);

  % Calculate gradient using backpropagation
  g_i = -back_prop_batch_gradient(train_set, target, nn_);
  d = g_i;

  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
    % Get weiths from neuro network structure    
    weigths = convert_neuronet_vw_to_w(nn_);
    
    % Functional to be minimized
    Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths + alpha*d, nn_))).^2);

    % Line search for alpha
    alpha = golden_search(0, 1, Jfunc, 1e-3);

    % Training method One-step Secant
    p = alpha*d;
    s = p;
    weigths = weigths + alpha*d;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);
    g_i1 = -back_prop_batch_gradient(train_set, target, nn_);

    q = g_i-g_i1;

    A = -(1 + (q'*q)/(s'*q))*((s'*-g_i)/(s'*q)) + (q'*-g_i)/(s'*q);
    B = (s'*-g_i)/(s'*q);

    d = g_i + A*s + B*q;
    
    % Calculation MSE error
    err_hist(:, it+1) = get_mse_error(input_sets, targets, nn_);

    mse_error = err_hist(1, it+1);
    it = it + 1;
    g_i = g_i1;
   
  end
  
end