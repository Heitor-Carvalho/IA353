function [nn_, err_hist, it] = batch_gradient_training(train_set, target, nn, train_par)
  % To do: add support to multiples outputs
  % To do: add function description

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
    J = back_prop_batch_gradient(train_set, target, nn_);

    % Normalize gradient
    J = J./norm(J);

    % GRAD optimization method with momentum
    delta_weigths = train_par.alpha*J - train_par.beta*J_past;
    J_past = delta_weigths;
    weigths =  weigths - delta_weigths;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);

    % Calculation MSE error
    mse_error = mean((target - neural_nete(train_set, nn_)).^2);

    it = it + 1;
    err_hist(it) = mse_error;

  end
  
end