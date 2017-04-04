function [grad, J] = back_prop_batch_gradient(train_set, target, nn)
  % back_prop_batch_gradient(train_set, target, nn) - Calculate the neural 
  % network gradient and Jacobian using the backpropagation algorithm. In 
  % this function only a one hiddnel layer neural network is considered.
  % Inputs:
  % train_set : Patterns used to train the neuro network
  % target    : Reference values to the patterns
  % Initialized neuro network structure:
  %   nn.v    : middle layer weights
  %   nn.w    : output layer weights
  %   nn.b    : neurons bias
  %   nn.func : neuron activation function 
  %   nn.diff : neuron activation function derivate
  %
  % Outputs:
  % grad      : Neural network weights gradient
  % J         : Neural network weights Jacobian
  %
  % To do : add support to multiple outputs

  samples_sz = size(train_set, 2);
  
  % Foward part - Neural network output
  [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn);  

  % Neuro network error
  error = target - nn_out;
  error = reshape(error, 1 , 1, samples_sz);
      
  % Output layer weights (Linear combiner)
  Jw = -mid_layer_func_out_bias;
  derror_dw = 2*repmat(error, 1, nn.mid_sz+1).*Jw;

  derror_dw_mean = mean(derror_dw, 3);
    
  % Middle layer weights
  w = repmat(nn.w(:, 2:end), 1, 1, samples_sz);
    
  Jv = -repmat(w, nn.in_sz+1, 1)                                         .* ...
        repmat(nn.diff(mid_layer_func_in), nn.in_sz+1, 1)                .* ...
        repmat(in_bias, 1, nn.mid_sz);

  derror_dv = 2*repmat(error, nn.in_sz+1, nn.mid_sz).*Jv;
  
  derror_dv_mean = mean(derror_dv, 3);
  
  grad_v = derror_dv_mean(:);
  grad_w = derror_dw_mean(:);

  grad = [grad_v; grad_w];
  
  Jw = reshape(Jw, nn.mid_sz+1, samples_sz)';
  Jv = reshape(Jv, (nn.in_sz+1)*nn.mid_sz, samples_sz)';
  J = [Jv Jw];
  
  
end