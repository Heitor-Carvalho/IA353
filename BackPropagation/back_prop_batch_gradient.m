function [nn_, err_hist, it] = back_prop_batch_gradient(train_set, target, nn, train_par)
  % To do : add support to multiple outputs
  % To do : add function description
  
  
  nn_ = nn;
  samples_sz = size(train_set, 2);
  mse_error = train_par.max_error;
  err_hist = zeros(1, train_par.max_it);
  it = 0;

  delta_w_past = 0;
  delta_v_past = 0;

  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
    % Foward part - Neural network output
    [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn_);  

    % Neuro network error
    error = target - nn_out;
    error = reshape(error, 1 , 1, samples_sz);
    mse_error = mean(error.^2);
      
    % Output layer weights (Linear combiner)
    derror_dw = -2*repmat(error, 1, nn_.mid_sz+1).*mid_layer_func_out_bias;
    derror_dw = mean(derror_dw, 3);
    
    % Middle layer weights
    w = repmat(nn_.w(:, 2:end), 1, 1, samples_sz);
    
    
    derror_dv = -2*repmat(error, nn_.in_sz+1, nn_.mid_sz)                            .* ...
                   repmat(w, nn_.in_sz+1, 1)                                         .* ...
                   repmat(nn_.diff(mid_layer_func_in), nn_.in_sz+1, 1)               .* ...
                   repmat(in_bias, 1, nn_.mid_sz);
    derror_dv = mean(derror_dv, 3);

    %Updating error
    derror_dw = derror_dw./norm(derror_dw);
    derror_dv = derror_dv./norm(derror_dv);

    delta_w = train_par.alpha*derror_dw + train_par.beta*delta_w_past;
    delta_v = train_par.alpha*derror_dv + train_par.beta*delta_v_past;
    
    nn_.w = nn_.w - delta_w;
    nn_.v = nn_.v - delta_v;
      
    delta_w_past = - delta_w;
    delta_v_past = - delta_v;
    
    it = it + 1;
    err_hist(it) = mse_error;

  end
  
end