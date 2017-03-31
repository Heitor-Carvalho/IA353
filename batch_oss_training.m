function [nn_, err_hist, it] = batch_oss_training(train_set, target, nn, train_par)
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
    J_ib = back_prop_batch_gradient(train_set, target, nn_);
    
    % Davidon-Fletcher-Powell    
    if(mod(it, length(J_ib)) == 0)
      d = -J_ib;
    end

    weigths = convert_neuronet_vw_to_w(nn_);
    
    % Functional to be minimized
    Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths + alpha*d, nn_))).^2);

    % Line search for alpha
    train_par.alpha = golden_search(0, 1, Jfunc, 1e-3);


    p = train_par.alpha*d;

    s = p;

    weigths = weigths + p;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);
    J_if = back_prop_batch_gradient(train_set, target, nn_);

    q = J_if-J_ib;

    A = -(1 + (q'*q)/(s'*q))*((s'*J_ib)/(s'*q)) + (q'*J_ib)/(s'*q);
    B = (s'*J_ib)/(s'*q);

    d = -J_ib + A*s + B*q;
   
    % Calculation MSE error
    mse_error = mean((target - neural_nete(train_set, nn_)).^2);

    it = it + 1;
    err_hist(it) = mse_error;
   
  end
  
end