function [nn_, err_hist, it] = back_prop_batch_dfp(train_set, target, nn, train_par)
  % To do: add support to multiples outputs
  % To do: add function description
  % Levenberg-Marquart  
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
  
  H = eye(size(J,1));
  
  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
    % Foward part - Neural network output
    [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn_);  

    % Neuro network error
    error = target - nn_out;
    error = reshape(error, 1 , 1, samples_sz);
    mean_error = mean(error, 3);
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
    
    Jv = derror_dv(:)';
    Jw = derror_dw(:)';
    J = [Jv Jw];
    keyboard
    
%    keyboard
   
    weigths = weigths - train_par.alpha*H*J;

    nn_.v = reshape(weigths(1:mid_layer_weigths_number), nn_.in_sz+1, nn_.mid_sz);
    nn_.w = weigths(mid_layer_weigths_number+1:end)';
    
    it = it + 1;
    mse_error
    err_hist(it) = mse_error;

  end
  
end