function [nn_, err_hist, it] = batch_lm_training(input_sets, targets, nn, train_par, mu)
  % To do: add support to multiples outputs
  % To do: add function description
  
  % Getting training and target sets:
  [train_set, input_sets] = get_train_set(input_sets);
  [target, targets] = get_train_target(targets);

  nn_ = nn;

  mse_error = train_par.max_error;
  err_hist = zeros(1, train_par.max_it);
  it = 0;

  delta_w_past = 0;
  delta_v_past = 0;
  
  samples_sz = size(train_set, 2);

  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
    % Foward part - Neural network output
    [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn_);  

    % Neuro network error
    error = target - nn_out;
    error = reshape(error, 1 , 1, samples_sz);

    mse_error = mean(error.^2);
      
    [~, J] = back_prop_batch_gradient(train_set, target, nn_);
  
    % Training method Levenberg-Marquart
    d2J = 2*pinv(J'*J + mu*eye(size(J,2)));
    deltaW = d2J*J'*reshape(error, samples_sz, 1);

    weigths = convert_neuronet_vw_to_w(nn_);

    Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths - alpha*deltaW, nn_))).^2);

    alpha =  golden_search(0, 10, Jfunc, 1e-4);
    
    weigths = weigths - alpha*deltaW;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);
  
    it = it + 1;
    err_hist(:, it) = get_mse_error(input_sets, targets, nn_);

    mse_error = err_hist(1, it);

  end
  
end