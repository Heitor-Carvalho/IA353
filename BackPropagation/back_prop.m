function [nn_, err_hist, it, oo] = back_prop(train_set, target, nn, train_par)
  % Falta multiplas saidas
 
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
  oo = [];
    for i = 1:size(train_set, 2)

      sample = train_set(:, i);
  
      % Foward part - Neural network output
      [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(sample, nn_);  
      oo = [oo; nn_out];
      % Neuro network error
      error = target(i)-nn_out;
      mse_error = mse_error + error.^2;
      
      % Output layer weights (Linear combiner)
      derror_dw = -2*repmat(error, 1, nn_.mid_sz+1).*mid_layer_func_out_bias;

      % Middle layer weights
      derror_dv = -2*repmat(error, nn_.in_sz+1, nn_.mid_sz)                 .* ...
                     repmat(nn_.w(:, 2:end), nn_.in_sz+1, 1)                .* ...
                     repmat(nn_.diff(mid_layer_func_in), nn_.in_sz+1, 1)   .* ...
                     repmat(in_bias, 1, nn_.mid_sz);


      %Updating error
%       derror_dw = derror_dw./norm(derror_dw);
%       derror_dv = derror_dv./norm(derror_dv);

      delta_w = - train_par.alpha*derror_dw;
      delta_v = - train_par.alpha*derror_dv;

      nn_.w = nn_.w + delta_w + train_par.beta*delta_w_past;
      nn_.v = nn_.v + delta_v + train_par.beta*delta_v_past;
      
      delta_w_past = delta_w;
      delta_v_past = delta_v;
    end
    
    it = it + 1;
    mse_error = mse_error/samples_sz
    err_hist(it) = mse_error;

  end
end