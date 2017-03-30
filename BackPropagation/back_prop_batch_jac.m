function [nn_, err_hist, it, oo] = back_prop_batch_jac(train_set, target, nn, train_par)
  % Falta multiplas entradas e multiplas saidas
  
  nn_ = nn;
  samples_sz = size(train_set, 2);
  mse_error = train_par.max_error;
  err_hist = zeros(1, train_par.max_it);
  it = 0;

  delta_w_past = 0;
  delta_v_past = 0;
  
  mid_layer_weigths_number = (nn_.in_sz+1)*nn_.mid_sz;
  output_layer_weigths_number = (nn_.mid_sz + 1)*nn_.out_sz;
  weitghs_number = mid_layer_weigths_number + output_layer_weigths_number;

  J = zeros(weitghs_number, 1);
  weigths = zeros(weitghs_number, 1);

  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
    % Foward part - Neural network output
    [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn_);  
    oo = nn_out;
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
    
    weigths(1:mid_layer_weigths_number) = nn_.v(:); 
    weigths(mid_layer_weigths_number+1:end) = nn_.w(:);

    J(1:mid_layer_weigths_number) = derror_dv(:); 
    J(mid_layer_weigths_number+1:end) = derror_dw(:);

    % Abstrac this!!!
    d2J = 0.2*2*pinv(J*J' + 0*eye(size(J,1)));
    weigths = weigths - train_par.alpha*d2J*J*mse_error;
    % end
    
    nn_.v = reshape(weigths(1:mid_layer_weigths_number), nn_.in_sz+1, nn_.mid_sz);
    nn_.w = weigths(mid_layer_weigths_number+1:end)';
    
    it = it + 1;
    mse_error
    err_hist(it) = mse_error;

  end
  
end