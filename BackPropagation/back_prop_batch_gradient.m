function [J, derror_dv, derror_dw] = back_prop_batch_gradient(train_set, target, nn)
  % To do : add support to multiple outputs
  % To do : add function description
  
  samples_sz = size(train_set, 2);
  
  % Foward part - Neural network output
  [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn);  

  % Neuro network error
  error = target - nn_out;
  error = reshape(error, 1 , 1, samples_sz);
      
  % Output layer weights (Linear combiner)
  derror_dw = -2*repmat(error, 1, nn.mid_sz+1).*mid_layer_func_out_bias;
  derror_dw_mean = mean(derror_dw, 3);
    
  % Middle layer weights
  w = repmat(nn.w(:, 2:end), 1, 1, samples_sz);
    
    
  derror_dv = -2*repmat(error, nn.in_sz+1, nn.mid_sz)                             .* ...
                 repmat(w, nn.in_sz+1, 1)                                         .* ...
                 repmat(nn.diff(mid_layer_func_in), nn.in_sz+1, 1)                .* ...
                 repmat(in_bias, 1, nn.mid_sz);
  derror_dv_mean = mean(derror_dv, 3);

  Jv = derror_dv_mean(:);
  Jw = derror_dw_mean(:);

  J = [Jv; Jw];
  
end