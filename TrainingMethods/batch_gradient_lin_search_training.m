function [nn_, err_hist, it] = batch_gradient_lin_search_training(input_sets, targets, nn, train_par, beta)
  % To do: add support to multiples outputs
  % To do: add function description

  % Getting training and target sets:
  [train_set, input_sets] = get_train_set(input_sets);
  [target, targets] = get_train_target(targets);

  nn_ = nn;

  mse_error = train_par.max_error;
  err_hist = zeros(1, train_par.max_it);
  it = 0;

  J_past = 0;
  
  samples_sz = size(train_set, 2);

  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
    
    % Get weiths from neuro network structure    
    weigths = convert_neuronet_vw_to_w(nn_);

    % Calculate gradient using backpropagation
    grad = back_prop_batch_gradient(train_set, target, nn_);

    % Normalize gradient
    grad = grad./norm(grad);

    % Functional to be minimized
    Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths - alpha*grad + beta*J_past, nn_))).^2);

    % Line search for alpha
    alpha = golden_search(0, 10, Jfunc, 1e-3);
        
    % Training method gradient
    delta_weigths = alpha*grad - beta*J_past;
    J_past = delta_weigths;
    weigths =  weigths - delta_weigths;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);

    % Calculation MSE error
    err_hist(:, it+1) = get_mse_error(input_sets, targets, nn_);

    mse_error = err_hist(1, it+1);
    it = it + 1;

  end
  
end
