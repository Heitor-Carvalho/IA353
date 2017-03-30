function [derror_dv, derror_dw, mse_error] = calculate_gradient(train_set, target, nn) 
  
  samples_sz = size(train_set, 2);
  mse_error = 0;
  
  % Foward part - Neural network output
  [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn);  

  % Neuro network error
  error = target - nn_out;
  error = reshape(error, 1 , 1, samples_sz);
  mse_error = mean(error.^2);
      
  % Output layer weights (Linear combiner)
  derror_dw = -2*repmat(error, 1, nn.mid_sz+1).*mid_layer_func_out_bias;
  derror_dw = mean(derror_dw, 3);
    
  % Middle layer weights
  w = repmat(nn.w(:, 2:end), 1, 1, samples_sz);
    
  derror_dv = -2*repmat(error, nn.in_sz+1, nn.mid_sz)                            .* ...
                 repmat(w, nn.in_sz+1, 1)                                         .* ...
                 repmat(nn.diff(mid_layer_func_in), nn.in_sz+1, 1)               .* ...
                 repmat(in_bias, 1, nn.mid_sz);
  derror_dv = mean(derror_dv, 3);

  

end